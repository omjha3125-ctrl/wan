import os
import sys
import re
import types
import pathlib

# Avoid audio backend issues on HF containers
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")


# -------------------------------------------------------------------
# 0) MUST preload mmgp.fp8_quanto_bridge stubs BEFORE importing mmgp/offload
# -------------------------------------------------------------------
def preload_mmgp_fp8_bridge_stubs():
    """
    mmgp.offload imports these names at import-time:
      from .fp8_quanto_bridge import convert_scaled_fp8_to_quanto, detect_safetensors_format
    On CPU/ZeroGPU or mismatched mmgp builds, that module may not exist or may be incomplete.

    This function ensures sys.modules contains mmgp.fp8_quanto_bridge with required symbols
    BEFORE any 'import mmgp' / 'import wgp' happens.
    """
    modname = "mmgp.fp8_quanto_bridge"
    bridge = sys.modules.get(modname)
    if bridge is None:
        bridge = types.ModuleType(modname)
        sys.modules[modname] = bridge

    # Safe no-op stubs
    def convert_scaled_fp8_to_quanto(tensor, *args, **kwargs):
        return tensor

    def detect_safetensors_format(*args, **kwargs):
        return None

    def load_quantized_model(*args, **kwargs):
        return None

    def enable_fp8_marlin_fallback(*args, **kwargs):
        return None

    # Inject/overwrite required attributes
    bridge.convert_scaled_fp8_to_quanto = convert_scaled_fp8_to_quanto
    bridge.detect_safetensors_format = detect_safetensors_format
    bridge.load_quantized_model = load_quantized_model
    bridge.enable_fp8_marlin_fallback = enable_fp8_marlin_fallback

    print("✅ Preloaded mmgp.fp8_quanto_bridge stubs (offload import safe).")


# -------------------------------------------------------------------
# 1) Patch mmgp.offload to remove torch.nn.Buffer(...) if present
# -------------------------------------------------------------------
def patch_mmgp_offload():
    """
    Some mmgp versions reference torch.nn.Buffer(...) in ways that break in certain envs.
    This patch edits the installed mmgp/offload.py in-place to remove torch.nn.Buffer wrappers.
    """
    try:
        import mmgp  # noqa: F401

        offload_path = pathlib.Path(mmgp.__file__).with_name("offload.py")
        if not offload_path.exists():
            print("⚠️ mmgp offload.py not found, skipping mmgp patch.")
            return

        text = offload_path.read_text()
        if "torch.nn.Buffer" in text:
            new_text = re.sub(
                r"torch\.nn\.Buffer\((.*?)\)",
                r"\1",
                text,
                flags=re.DOTALL,
            )
            if new_text != text:
                offload_path.write_text(new_text)
                print("✅ Patched mmgp.offload (removed torch.nn.Buffer).")
            else:
                print("✅ mmgp.offload contained torch.nn.Buffer but no change was needed.")
        else:
            print("✅ torch.nn.Buffer not found in mmgp.offload, no patch necessary.")
    except Exception as e:
        print(f"⚠️ mmgp offload patch failed: {e}")


# -------------------------------------------------------------------
# 2) Patch spaces.zero.wrappers pickling issue (GradioPartialContext)
# -------------------------------------------------------------------
def patch_spaces_zero_pickling():
    """
    Fix ZeroGPU crash:
      _pickle.PicklingError: cannot pickle '_thread.lock' object
    caused by pickling GradioPartialContext.get() into multiprocessing queue.

    We patch spaces.zero.wrappers in-place:
      worker.arg_queue.put(((args, kwargs), GradioPartialContext.get()))
      -> worker.arg_queue.put(((args, kwargs), None))
    """
    try:
        import spaces.zero.wrappers as wrappers  # noqa: F401

        wrappers_path = pathlib.Path(wrappers.__file__)
        text = wrappers_path.read_text()

        target = "worker.arg_queue.put(((args, kwargs), GradioPartialContext.get()))"
        if target in text:
            new_text = text.replace(
                "worker.arg_queue.put(((args, kwargs), GradioPartialContext.get()))",
                "worker.arg_queue.put(((args, kwargs), None))",
            )
            if new_text != text:
                wrappers_path.write_text(new_text)
                print("✅ Patched spaces.zero.wrappers to avoid pickling GradioPartialContext.")
            else:
                print("✅ spaces.zero.wrappers pattern unchanged (no diff).")
        else:
            print("✅ spaces.zero.wrappers already patched or pattern not found.")
    except Exception as e:
        print(f"⚠️ spaces.zero.wrappers patch failed: {e}")


# -------------------------------------------------------------------
# 3) Clamp slider preprocess to avoid "Value 0 is less than minimum value 1.0"
# -------------------------------------------------------------------
def patch_gradio_slider_clamp():
    """
    Gradio can error if slider sends 0 while minimum is 1. Clamp values in preprocess.
    """
    try:
        from gradio.components import Slider

        orig = Slider.preprocess

        def clamped(self, x):
            try:
                min_val = getattr(self, "minimum", None)
                max_val = getattr(self, "maximum", None)
                if min_val is not None and x is not None and x < min_val:
                    print(f"[Slider clamp RT] value {x} < min {min_val}, clamping.")
                    x = min_val
                if max_val is not None and x is not None and x > max_val:
                    print(f"[Slider clamp RT] value {x} > max {max_val}, clamping.")
                    x = max_val
            except Exception:
                pass
            return orig(self, x)

        Slider.preprocess = clamped
        print("✅ Runtime slider clamp patch applied.")
    except Exception as e:
        print(f"⚠️ Runtime slider clamp patch failed: {e}")


# -------------------------------------------------------------------
# Apply patches BEFORE importing wgp (which imports mmgp.offload)
# -------------------------------------------------------------------
preload_mmgp_fp8_bridge_stubs()
patch_mmgp_offload()
patch_spaces_zero_pickling()
patch_gradio_slider_clamp()

# Ensure Wan2GP starts in i2v mode like you want
sys.argv = ["wgp.py", "--i2v"]

# Import Wan2GP AFTER patches
import wgp  # noqa: E402

# Hugging Face expects a top-level 'demo' Blocks
try:
    demo = wgp.create_ui()
    print("✅ Built Gradio Blocks via wgp.create_ui().")
except Exception as e:
    # If create_ui fails, bubble up with useful message
    raise RuntimeError(f"Failed to build UI via wgp.create_ui(): {e}") from e


# Local run (HF ignores this; it imports demo)
if __name__ == "__main__":
    demo.queue().launch(server_name="0.0.0.0", server_port=7860)
