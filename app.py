import os
import sys
import re
import types
import pathlib
import importlib
import multiprocessing as mp
from pathlib import Path

# ---- MUST be first: spawn before anything that might touch CUDA ----
try:
    mp.set_start_method("spawn", force=True)
    print("✅ multiprocessing start method set to spawn")
except RuntimeError:
    pass

# ---- MUST be before torch/mmgp/wgp: spaces must be imported early on ZeroGPU ----
import spaces  # noqa: F401


def patch_spaces_zero_wrappers():
    """
    Fix ZeroGPU worker init errors caused by forking with CUDA:
      RuntimeError: Cannot re-initialize CUDA in forked subprocess
    Also keep the pickling fix for GradioPartialContext.
    """
    try:
        import spaces.zero.wrappers as wrappers  # noqa: F401

        wrappers_path = Path(wrappers.__file__)
        txt = wrappers_path.read_text(encoding="utf-8")

        changed = False

        # Force spawn instead of fork if hardcoded in wrappers
        for old, new in [
            ("get_context('fork')", "get_context('spawn')"),
            ('get_context("fork")', 'get_context("spawn")'),
        ]:
            if old in txt:
                txt = txt.replace(old, new)
                changed = True

        # Keep your pickling fix too (if present)
        old_pick = "worker.arg_queue.put(((args, kwargs), GradioPartialContext.get()))"
        new_pick = "worker.arg_queue.put(((args, kwargs), None))"
        if old_pick in txt:
            txt = txt.replace(old_pick, new_pick)
            changed = True

        if changed:
            wrappers_path.write_text(txt, encoding="utf-8")
            print("✅ Patched spaces.zero.wrappers on disk (fork→spawn + pickling fix).")
            importlib.reload(wrappers)
            print("✅ Reloaded spaces.zero.wrappers after patch.")
        else:
            print("✅ spaces.zero.wrappers: patterns not found or already patched.")

        # Runtime override (belt + suspenders)
        ctx = mp.get_context("spawn")
        for attr in ("ctx", "mp_ctx", "_mp_ctx", "MP_CTX"):
            if hasattr(wrappers, attr):
                setattr(wrappers, attr, ctx)

        # Override Process/Queue symbols if wrappers exposes them
        for attr, val in [
            ("Process", ctx.Process),
            ("Queue", ctx.Queue),
            ("SimpleQueue", ctx.SimpleQueue),
        ]:
            if hasattr(wrappers, attr):
                setattr(wrappers, attr, val)

        print("✅ Runtime override: wrappers now uses spawn context.")
    except Exception as e:
        print(f"⚠️ patch_spaces_zero_wrappers failed: {e}")


def preload_mmgp_fp8_bridge_stubs():
    """
    mmgp.offload imports these at import-time:
      from .fp8_quanto_bridge import convert_scaled_fp8_to_quanto, detect_safetensors_format
    Ensure they're present BEFORE importing mmgp/offload.
    """
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


def patch_gradio_slider_clamp():
    """
    Prevent Gradio slider crash:
      Value 0 is less than minimum value 1.0
    """
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


def ensure_wgp_plugin_app(wgp_module):
    """
    wgp.create_ui expects a global `app` with plugin methods.
    Inject real WAN2GPApplication if available, otherwise a no-op dummy.
    """
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


# Env hygiene
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

# 1) Patch spaces wrappers FIRST (fork→spawn + pickling)
patch_spaces_zero_wrappers()

# 2) Preload mmgp bridge stubs before any mmgp import
preload_mmgp_fp8_bridge_stubs()

# 3) Patch mmgp.offload (now safe)
patch_mmgp_offload()

# 4) Patch slider clamp early
patch_gradio_slider_clamp()

# Ensure Wan2GP runs with --i2v
sys.argv = ["wgp.py", "--i2v"]

# Import wgp only AFTER spaces + patches
import wgp  # noqa: E402

# Inject plugin manager into wgp namespace
ensure_wgp_plugin_app(wgp)

# Build HF-facing app
demo = wgp.create_ui()
print("✅ Built Gradio Blocks via wgp.create_ui().")

# Local execution (HF ignores this; it imports `demo`)
if __name__ == "__main__":
    demo.queue().launch(server_name="0.0.0.0", server_port=7860)
