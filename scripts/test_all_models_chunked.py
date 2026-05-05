"""Quick ctor check for halo models with use_chunked_ce."""
import sys
sys.path.insert(0, '.')

models_to_test = [
    ("models.vidar_halo", "VidarHalo"),
    ("models.fenrir_halo", "FenrirHalo"),
    ("models.tyr_halo", "TyrHalo"),
    ("models.baldr_halo", "BaldrHalo"),
    ("models.chimera_halo", "ChimeraHalo"),
    ("models.odin_halo", "OdinHalo"),
]

for module_path, class_name in models_to_test:
    try:
        mod = __import__(module_path, fromlist=[class_name])
        cls = getattr(mod, class_name)
        m = cls(use_chunked_ce=True)
        has_flag = getattr(m, 'use_chunked_ce', False)
        has_softcap = hasattr(m, 'logit_softcap')
        print(f"  {class_name:20s} use_chunked_ce={has_flag} logit_softcap={has_softcap}")
    except Exception as e:
        print(f"  {class_name:20s} FAIL: {type(e).__name__}: {str(e)[:100]}")
