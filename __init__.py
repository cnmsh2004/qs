from .node import QsLlmNode
from .preset_node import QsPresetConfigNode

NODE_CLASS_MAPPINGS = {
    "Qs-LLM": QsLlmNode,
    "Qs-PresetConfig": QsPresetConfigNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Qs-LLM": "Qs-LLM",
    "Qs-PresetConfig": "Qs-预设配置",
}