# app.py — ZeroGPU-safe bootstrap for Wan2GP (Python 3.10)

import os
import sys
import re
import types
import pathlib
import importlib
import multiprocessing as mp
from pathlib import Path

# 0) Force spawn as early as possible
try:
    mp.set_start_method("spawn", force=True)
    print("✅ multiprocessing start method set to spawn (forced)")
except RuntimeError:
    pass

# 0.1) Make spawn able to pickle nested functions used by spaces.zero.wrappers
def patch_multiprocessing_cloudpickle():
    try:
        import cloudpickle  # <-- add to requirements.txt
        import multiprocessing.reduction as reduction

        # Make multiprocessing use cloudpickle for spawn payloads
        reduction.ForkingPickler = cloudpickle.CloudPickler  # type: ignore[attr-defined]

        def _cloud_dump(obj, file, protocol=None):
            cloudpickle.dump(obj, file, protocol=protocol)

        reduction.dump = _cloud_dump  # type: ignore[assignment]

        print("✅ Patched multiprocessing pickler to cloudpickle (spawn can pickle nested functions).")
    except Exception as e:
        raise RuntimeError(
            "cloudpickle is required to run ZeroGPU with spawn here. "
            "Add `cloudpickle` to requirements.txt and rebuild."
        ) from e


patch_multiprocessing_cloudpickle()

# 1) Import spaces BEFORE torch/mmgp/wgp (ZeroGPU requirement)
import spaces  # noqa: F401


def patch_spaces_zero_wrappers_on_disk():
    """
    Replace fork->spawn and avoid GradioPartialContext pickling (if patterns match).
    """
    try:
        import spaces.zero.wrappers as wrappers

        p = Path(wrappers.__file__)
        txt = p.read_text(encoding="utf-8")
        changed = False

        for old, new in [
            ("get_context('fork')", "get_context('spawn')"),
            ('get_context("fork")', 'get_context("spawn")'),
        ]:
            if old in txt:
                txt = txt.replace(old, new)
                changed = True

        old_pick = "worker.arg_queue.put(((args, kwargs), GradioPartialContext.get()))"
        new_pick = "worker.arg_queue.put(((args, kwargs), None))"
        if old_pick in txt:
            txt = txt.replace(old_pick, new_pick)
            changed = True

        if changed:
            p.write_text(txt, encoding="utf-8")
            print("✅ Patched spaces.zero.wrappers on disk (fork→spawn + pickling fix).")
            importlib.reload(wrappers)
            print("✅ Reloaded spaces.zero.wrappers after patch.")
        else:
            print("✅ spaces.zero.wrappers: no disk patch needed (patterns not found or already patched).")
    except Exception as e:
        print(f"⚠️ patch_spaces_zero_wrappers_on_disk failed: {e}")


def patch_spaces_zero_wrappers_runtime():
    """
    Force wrappers to use spawn context and avoid pickling GradioPartialContext at runtime.
    """
    try:
        import spaces.zero.wrappers as wrappers

        ctx = mp.get_context("spawn")

        for attr in ("Process", "Queue", "SimpleQueue"):
            if hasattr(wrappers, attr):
                setattr(wrappers, attr, getattr(ctx, attr))

        for attr in ("ctx", "mp_ctx", "_mp_ctx", "MP_CTX"):
            if hasattr(wrappers, attr):
                setattr(wrappers, attr, ctx)

        if hasattr(wrappers, "GradioPartialContext") and hasattr(wrappers.GradioPartialContext, "get"):
            wrappers.GradioPartialContext.get = staticmethod(lambda: None)

        print("✅ Patched spaces.zero.wrappers runtime (spawn + no GradioPartialContext pickling).")
    except Exception as e:
        print(f"⚠️ patch_spaces_zero_wrappers_runtime failed: {e}")


def preload_mmgp_fp8_bridge_stubs():
    """
    Ensure mmgp.offload import doesn't crash on fp8_quanto_bridge missing symbols.
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
    Prevent Gradio slider crash: Value 0 < min 1.0
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
    wgp.create_ui expects global `app` with plugin methods.
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


# ---- Startup order ----
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

# Patch spaces wrappers BEFORE importing wgp (and before GPU job runs)
patch_spaces_zero_wrappers_on_disk()
patch_spaces_zero_wrappers_runtime()

# mmgp stubs before mmgp import
preload_mmgp_fp8_bridge_stubs()
patch_mmgp_offload()

# slider clamp
patch_gradio_slider_clamp()

# Force Wan2GP i2v mode
sys.argv = ["wgp.py", "--i2v"]

# Import wgp only after the above
import wgp  # noqa: E402

ensure_wgp_plugin_app(wgp)

demo = wgp.create_ui()
print("✅ Built Gradio Blocks via wgp.create_ui().")

if __name__ == "__main__":
    demo.queue().launch(server_name="0.0.0.0", server_port=7860, ssr_mode=False)
