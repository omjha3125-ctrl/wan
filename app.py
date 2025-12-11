import os
import sys
import re
import types
import runpy
import importlib
import inspect
import pathlib

def patch_mmgp_offload():
    """Patch mmgp.offload to remove torch.nn.Buffer usage"""
    try:
        import mmgp
        offload_path = pathlib.Path(mmgp.__file__).with_name("offload.py")
        text = offload_path.read_text()
        if "torch.nn.Buffer(" in text:
            # naive but safe: remove torch.nn.Buffer wrapper
            text_new = re.sub(
                r"torch\.nn\.Buffer\((.*?)\)",
                r"\1",
                text,
                flags=re.DOTALL,
            )
            if text_new != text:
                offload_path.write_text(text_new)
                print("[OK] Patched mmgp.offload (removed torch.nn.Buffer).")
            else:
                print("[OK] mmgp.offload had torch.nn.Buffer string but no changes were needed.")
        else:
            print("[OK] torch.nn.Buffer not found in mmgp.offload, no patch required.")
    except Exception as e:
        print(f"[WARNING] mmgp offload patch failed: {e}")

def runtime_patch_mmgp_bridge():
    """Fix ZeroGPU / CPU-only startup by mocking fp8_quanto_bridge if absent"""
    try:
        import mmgp
        if hasattr(mmgp, "fp8_quanto_bridge"):
            print("[OK] mmgp.fp8_quanto_bridge already present.")
            return

        class _DummyBridge:
            def __getattr__(self, _):
                return self
            def __call__(self, *a, **k):
                return self
            def __bool__(self):
                return False
            # Add the missing functions that mmgp.offload.py tries to import
            def convert_scaled_fp8_to_quanto(self, *args, **kwargs):
                return None
            def detect_safetensors_format(self, *args, **kwargs):
                return None

        dummy = _DummyBridge()
        mmgp.fp8_quanto_bridge = dummy

        fake_mod = types.ModuleType("mmgp.fp8_quanto_bridge")
        fake_mod.load_quantized_model = dummy
        fake_mod.enable_fp8_marlin_fallback = dummy
        fake_mod.convert_scaled_fp8_to_quanto = dummy.convert_scaled_fp8_to_quanto
        fake_mod.detect_safetensors_format = dummy.detect_safetensors_format
        sys.modules["mmgp.fp8_quanto_bridge"] = fake_mod

        print("[OK] Injected dummy mmgp.fp8_quanto_bridge.")
    except Exception as e:
        print(f"[WARNING] Runtime mmgp bridge patch failed: {e}")

def patch_spaces_zero_pickling():
    """Patch spaces.zero.wrappers to avoid pickling GradioPartialContext"""
    try:
        import spaces.zero.wrappers as wrappers
        wrappers_path = pathlib.Path(wrappers.__file__)
        text = wrappers_path.read_text()

        # Original line typically:
        # worker.arg_queue.put(((args, kwargs), GradioPartialContext.get()))
        pattern = "worker.arg_queue.put(((args, kwargs), GradioPartialContext.get()))"

        if pattern in text:
            text_new = text.replace(
                "worker.arg_queue.put(((args, kwargs), GradioPartialContext.get()))",
                "worker.arg_queue.put(((args, kwargs), None))",
            )
            if text_new != text:
                wrappers_path.write_text(text_new)
                print("[OK] Patched spaces.zero.wrappers to avoid pickling GradioPartialContext.")
            else:
                print("[OK] spaces.zero.wrappers pattern unchanged (no diff).")
        else:
            print("[OK] spaces.zero.wrappers already patched or pattern not found.")
    except Exception as e:
        print(f"[WARNING] spaces.zero.wrappers patch failed: {e}")

def patch_gradio_slider_clamp():
    """Patch Gradio Slider to clamp values to min/max before validation"""
    try:
        import gradio as gr
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
        print("[OK] Runtime slider clamp patch applied.")
    except Exception as e:
        print(f"[WARNING] Runtime slider clamp patch failed: {e}")

def main():
    repo_dir = os.path.dirname(os.path.abspath(__file__))
    if repo_dir not in sys.path:
        sys.path.insert(0, repo_dir)

    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

    # Apply runtime patches BEFORE importing wgp.py
    patch_mmgp_offload()
    runtime_patch_mmgp_bridge()
    patch_spaces_zero_pickling()
    patch_gradio_slider_clamp()

    # Prepare to run Wan2GP in i2v mode
    sys.argv = ["wgp.py", "--i2v"]
    print("[INFO] Wan2GP ready to launch: python wgp.py --i2v via app.main()")
    # NOTE: Do NOT call runpy.run_path() here in this editing step.
    # The host environment will invoke main() when appropriate.

if __name__ == "__main__":
    main()