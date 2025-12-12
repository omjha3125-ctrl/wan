import os
import sys
import re
import types
import pathlib
import multiprocessing as mp

try:
    mp.set_start_method("spawn", force=True)
    print("✅ multiprocessing start method set to spawn")
except RuntimeError:
    # already set by environment
    pass

# IMPORTANT: import spaces BEFORE any torch/mmgp/wgp import (ZeroGPU requirement)
import spaces  # noqa: F401

os.environ.setdefault("SDL_AUDIODRIVER", "dummy")


# -------------------------------------------------------------------
# 0) Preload mmgp.fp8_quanto_bridge stubs BEFORE importing mmgp/offload
# -------------------------------------------------------------------
def preload_mmgp_fp8_bridge_stubs():
    modname = "mmgp.fp8_quanto_bridge"
    bridge = sys.modules.get(modname)
    if bridge is None:
        bridge = types.ModuleType(modname)
        sys.modules[modname] = bridge

    def convert_scaled_fp8_to_quanto(tensor, *args, **kwargs):
        return tensor

    def detect_safetensors_format(*args, **kwargs):
        return None

    def load_quantized_model(*args, **kwargs):
        return None

    def enable_fp8_marlin_fallback(*args, **kwargs):
        return None

    bridge.convert_scaled_fp8_to_quanto = convert_scaled_fp8_to_quanto
    bridge.detect_safetensors_format = detect_safetensors_format
    bridge.load_quantized_model = load_quantized_model
    bridge.enable_fp8_marlin_fallback = enable_fp8_marlin_fallback

    print("✅ Preloaded mmgp.fp8_quanto_bridge stubs (offload import safe).")


# -------------------------------------------------------------------
# 1) Patch mmgp.offload to remove torch.nn.Buffer(...) if present
# -------------------------------------------------------------------
def patch_mmgp_offload():
    try:
        import mmgp  # noqa: F401

        offload_path = pathlib.Path(mmgp.__file__).with_name("offload.py")
        if not offload_path.exists():
            print("⚠️ mmgp offload.py not found, skipping.")
            return

        text = offload_path.read_text()
        if "torch.nn.Buffer" in text:
            new_text = re.sub(r"torch\.nn\.Buffer\((.*?)\)", r"\1", text, flags=re.DOTALL)
            if new_text != text:
                offload_path.write_text(new_text)
                print("✅ Patched mmgp.offload (removed torch.nn.Buffer).")
            else:
                print("✅ mmgp.offload: torch.nn.Buffer present but no change needed.")
        else:
            print("✅ torch.nn.Buffer not found in mmgp.offload, no patch necessary.")
    except Exception as e:
        print(f"⚠️ mmgp offload patch failed: {e}")


# -------------------------------------------------------------------
# 2) Patch spaces.zero.wrappers pickling issue (GradioPartialContext)
# -------------------------------------------------------------------
def patch_spaces_zero_pickling():
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
            wrappers_path.write_text(new_text)
            print("✅ Patched spaces.zero.wrappers to avoid pickling GradioPartialContext.")
        else:
            print("✅ spaces.zero.wrappers already patched or pattern not found.")
    except Exception as e:
        print(f"⚠️ spaces.zero.wrappers patch failed: {e}")


# -------------------------------------------------------------------
# 3) Clamp Gradio slider preprocess to avoid min-bound crash
# -------------------------------------------------------------------
def patch_gradio_slider_clamp():
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


# Apply patches BEFORE importing wgp
preload_mmgp_fp8_bridge_stubs()
patch_mmgp_offload()
patch_spaces_zero_pickling()
patch_gradio_slider_clamp()

# Force Wan2GP i2v mode like you want
sys.argv = ["wgp.py", "--i2v"]

import wgp  # noqa: E402


def _ensure_wgp_plugin_app():
    if getattr(wgp, "app", None) is not None:
        print("[Plugin] wgp.app already present.")
        return

    try:
        from shared.utils.plugins import WAN2GPApplication

        wgp.app = WAN2GPApplication()
        print("[Plugin] WAN2GPApplication injected by app.py.")
    except Exception as e:
        class _DummyPluginApp:
            def initialize_plugins(self, globals_dict):
                print("[Plugin] Dummy initialize_plugins (no-op).")

            def run_component_insertion(self, locals_dict):
                print("[Plugin] Dummy run_component_insertion (no-op).")

            def setup_ui_tabs(self, *args, **kwargs):
                print("[Plugin] Dummy setup_ui_tabs (no-op).")

            def get_tab_order(self):
                return []

        wgp.app = _DummyPluginApp()
        print(f"[Plugin] Using DummyPluginApp (plugins disabled): {e}")


_ensure_wgp_plugin_app()

demo = wgp.create_ui()
print("✅ Built Gradio Blocks via wgp.create_ui().")

if __name__ == "__main__":
    demo.queue().launch(server_name="0.0.0.0", server_port=7860)
