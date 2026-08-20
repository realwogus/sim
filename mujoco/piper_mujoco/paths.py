from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PIPER_MODEL_DIR = PROJECT_ROOT / "models" / "mujoco_menagerie" / "agilex_piper"
PIPER_XML = PIPER_MODEL_DIR / "piper.xml"
PIPER_SCENE_XML = PIPER_MODEL_DIR / "scene.xml"
GR00T_SCENE_XML = PROJECT_ROOT / "models" / "scenes" / "piper_gr00t.xml"
SIMULATION_CONFIG = PROJECT_ROOT / "configs" / "simulation.yaml"
TRAINING_CONFIG = PROJECT_ROOT / "configs" / "training.yaml"
