import os
import gc

from comfy_api.latest import io, ComfyExtension
import folder_paths
try:
    import torch
except ImportError:
    torch = None

from llama_cpp import Llama


# --- Model directory ---
LLM_DIR = os.path.join(folder_paths.models_dir, "LLM")


def list_gguf_models():
    if not os.path.isdir(LLM_DIR):
        return []
    return sorted(f for f in os.listdir(LLM_DIR) if f.lower().endswith(".gguf"))


def build_prompt(task_type: str, text: str) -> str:
    if task_type == "prompt_enhance":
        return (
            "You rewrite prompts for image generation. Improve clarity, detail, and structure "
            "without changing the core intent.\n\n"
            f"Original:\n{text}\n\nImproved:"
        )
    if task_type == "caption_refine":
        return (
            "You refine image captions. Make them clearer, more descriptive, and grammatically correct "
            "without inventing details.\n\n"
            f"Original:\n{text}\n\nRefined:"
        )
    return text


class LlamaCppLLM(io.ComfyNode):

    @classmethod
    def define_schema(cls):
        models = list_gguf_models() or ["<no_gguf_found>"]

        return io.Schema(
            node_id="LlamaCppLLM",
            display_name="Llama.cpp LLM",
            category="LLM/llama.cpp",

            inputs=[
                io.String.Input(
                    id="text",
                    display_name="Input Text",
                    multiline=True,
                    default=""
                ),
                io.Combo.Input(
                    id="model_name",
                    display_name="Model",
                    options=["None"] + models,
                    default=None
                ),
                io.Combo.Input(
                    id="task_type",
                    display_name="Task Type",
                    options=["prompt_enhance", "caption_refine", "raw_completion"],
                    default="prompt_enhance"
                ),
                io.Int.Input(
                    id="max_tokens",
                    display_name="Max Tokens",
                    default=256,
                    min=1,
                    max=2048
                ),
                io.Float.Input(
                    id="temperature",
                    display_name="Temperature",
                    default=0.7,
                    min=0.0,
                    max=2.0,
                    step=0.05
                ),
                io.Int.Input(
                    id="n_ctx",
                    display_name="Context Length",
                    default=2048,
                    min=256,
                    max=8192
                ),
                io.Int.Input(
                    id="n_gpu_layers",
                    display_name="GPU Layers (-1 = all)",
                    default=-1,
                    min=-1,
                    max=200
                ),
                io.Boolean.Input(
                    id="clear_cuda_cache",
                    display_name="Clear CUDA Cache After Run",
                    default=True
                ),
            ],

            outputs=[
                io.String.Output(
                    id="output_text",
                    display_name="Output Text"
                )
            ]
        )

    @classmethod
    def execute(
        cls,
        text,
        model_name,
        task_type,
        max_tokens,
        temperature,
        n_ctx,
        n_gpu_layers,
        clear_cuda_cache,
    ):
        if model_name == "<no_gguf_found>":
            return io.NodeOutput(output_text="[LlamaCppLLM] No GGUF models found in models/LLM")

        model_path = os.path.join(LLM_DIR, model_name)
        if not os.path.isfile(model_path):
            return io.NodeOutput(output_text=f"[LlamaCppLLM] Model not found: {model_path}")

        prompt = build_prompt(task_type, text)

        llm = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            logits_all=False,
            verbose=False,
        )

        try:
            result = llm(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                stop=["</s>", "###"],
            )
            out_text = result["choices"][0]["text"].strip()

        finally:
            del llm
            gc.collect()
            if torch and clear_cuda_cache and torch.cuda.is_available():
                torch.cuda.empty_cache()

        return io.NodeOutput(output_text=out_text)


class LlamaCppExtension(ComfyExtension):
    async def get_node_list(self):
        return [LlamaCppLLM]


async def comfy_entrypoint():
    return LlamaCppExtension()
