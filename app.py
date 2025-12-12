# app.py
# ZeroGPU-safe bootstrap for Wan2GP

import multiprocessing as mp
import multiprocessing.context as mp_context

# ----------------------------
# 0) HARD-FORCE spawn (EARLY)
# ----------------------------
# Patch get_context so even if some library asks for "fork", it gets "spawn".
_orig_get_context = mp_context.get_context


def _patched_get_context(method=None):
    if method == "fork":
        method = "spawn"
    return _orig_get_context(method)


mp_context.get_context = _patched_get_context
mp.get_context = _patched_get_context

try:
    mp.set_start_method("spawn", force=True)
    print("✅ multiprocessing start method set to spawn (forced)")
except RuntimeError:
    # Already set by environment
    pass

# ----------------------------
# 1) Import spaces BEFORE CUDA/torch touches anything
# ----------------------------
import spaces  # noqa: F401

import os
import sys
import re
import types
import pathlib
import importlib


def patch_spaces_runtime():
    """
    Ensure spaces zero worker uses spawn and avoid pickling GradioPartialContext.
    (Even if wrappers imported `get_context` directly, we replace it here too.)
    """
    try:
        import spaces.zero.wrappers as wrappers

        # Force wrappers-level get_context to our patched one
        if hasattr(wrappers, "get_context"):
            wrappers.get_context = _patched_get_context

        # Avoid pickling GradioPartialContext (safe default)
        if hasattr(wrappers, "GradioPartialContext") and hasattr(wrappers.GradioPartialContext, "get"):
            wrappers.GradioPartialContext.get = staticmethod(lambda: None)

        # In case wrappers cached a ctx/Process/Queue already, override them
        ctx = mp.get_context("spawn")
        for name in ("Process", "Queue", "SimpleQueue"):
            if hasattr(wrappers, name):
                setattr(wrappers, name, getattr(ctx, name))

        print("✅ Patched spaces runtime (spawn + no GradioPartialContext pickling).")
    except Exception as e:
        print(f"⚠️ patch_spaces_runtime failed: {e}")


# ----------------------------
# 2) mmgp bridge stubs (must exist BEFORE mmgp.offload import)
# ----------------------------
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


def patch_mmgp_offload():
    """
    Remove torch.nn.Buffer(...) wrapper if present in installed mmgp/offload.py
    """
    try:
        import mmgp  # noqa: F401

        offload_path = pathlib.Path(mmgp.__file__).with_name("offload.py")
        if not offload_path.exists():
            print("⚠️ mmgp offload.py not found, skipping.")
            return

        text = offload_path.read_text(encoding="utf-8")
        if "torch.nn.Buffer" in text:
            new_text = re.sub(r"torch\.nn\.Buffer\((.*?)\)", r"\1", text, flags=re.DOTALL)
            if new_text != text:
                offload_path.write_text(new_text, encoding="utf-8")
                print("✅ Patched mmgp.offload (removed torch.nn.Buffer).")
            else:
                print("✅ mmgp.offload: torch.nn.Buffer present but no change needed.")
        else:
            print("✅ torch.nn.Buffer not found in mmgp.offload, no patch necessary.")
    except Exception as e:
        print(f"⚠️ mmgp offload patch failed: {e}")


# ----------------------------
# 3) Gradio slider clamp (stop Value 0 < min 1 crash)
# ----------------------------
def patch_gradio_slider_clamp():
    try:
        from gradio.components import Slider

        orig = Slider.preprocess

        def clamped(self, x):
            try:
                mn = getattr(self, "minimum", None)
                mx = getattr(self, "maximum", None)
                if mn is not None and x is not None and x < mn:
                    print(f"[Slider clamp] value {x} < min {mn}, clamping.")
                    x = mn
                if mx is not None and x is not None and x > mx:
                    print(f"[Slider clamp] value {x} > max {mx}, clamping.")
                    x = mx
            except Exception:
                pass
            return orig(self, x)

        Slider.preprocess = clamped
        print("✅ Runtime slider clamp patch applied.")
    except Exception as e:
        print(f"⚠️ Runtime slider clamp patch failed: {e}")


# ----------------------------
# 4) Ensure plugin app exists for wgp.create_ui()
# ----------------------------
def ensure_wgp_plugin_app(wgp_module):
    if getattr(wgp_module, "app", None) is not None:
        print("[Plugin] wgp.app already present.")
        return

    try:
        from shared.utils.plugins import WAN2GPApplication

        wgp_module.app = WAN2GPApplication()
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

        wgp_module.app = _DummyPluginApp()
        print(f"[Plugin] Using DummyPluginApp (plugins disabled): {e}")


# ----------------------------
# Startup sequence (order matters)
# ----------------------------
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

patch_spaces_runtime()
preload_mmgp_fp8_bridge_stubs()
patch_mmgp_offload()
patch_gradio_slider_clamp()

# Force Wan2GP i2v mode
sys.argv = ["wgp.py", "--i2v"]

# Import wgp AFTER spaces + multiprocessing patches
import wgp  # noqa: E402

ensure_wgp_plugin_app(wgp)

demo = wgp.create_ui()
print("✅ Built Gradio Blocks via wgp.create_ui().")

if __name__ == "__main__":
    demo.queue().launch(server_name="0.0.0.0", server_port=7860, ssr_mode=False)
