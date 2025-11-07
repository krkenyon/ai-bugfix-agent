import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_MODEL_NAME = "Qwen/Qwen3-0.6B"


class BugFixModel:
    """
    Thin wrapper around a causal language model for code fixing.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        device: str | None = None,
        max_input_tokens: int = 512,
    ):
        if device is None:
            # use GPU if available - much faster for inference
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = device
        self.model_name = model_name
        self.max_input_tokens = max_input_tokens

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        dtype = torch.float16 if device == "cuda" else torch.float32

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=dtype,
        ).to(device)

    def _looks_like_python_module(self, text: str) -> bool:
        lines = [l for l in text.splitlines() if l.strip()]
        if not lines:
            return False
        first = lines[0].lstrip()
        return first.startswith(("def ", "class ", "import ", "from "))


    def _clean_generation(self, raw: str) -> str:
        text = raw.strip()

        # 1) Remove any explicit <think>...</think> blocks (if closed)
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)

        # 2) Remove stray opening/closing <think> tags and the line they are on
        # (handles the case where model only outputs `<think>` with no closing tag)
        text = re.sub(r"^<think>\s*\n?", "", text, flags=re.MULTILINE)
        text = text.replace("</think>", "")

        text = text.strip()

        # 3) Prefer fenced code blocks if present
        fence = re.search(r"```(?:python)?\s*([\s\S]*?)```", text)
        if fence:
            return fence.group(1).strip()

        # 4) If there's any line that starts like real Python code,
        #    cut everything before the first such line.
        m = re.search(r"(?m)^\s*(def|class|from|import)\s", text)
        if m:
            text = text[m.start():]

        # 5) Optionally trim trailing obvious prose:
        #    stop when lines stop looking like code.
        lines = text.splitlines()
        code_lines = []
        started = False

        for line in lines:
            stripped = line.strip()

            if not started:
                if stripped.startswith(("def ", "class ", "import ", "from ")):
                    started = True
                    code_lines.append(line)
                # else still skipping
            else:
                if stripped == "":
                    code_lines.append(line)
                    continue

                if (
                    stripped.startswith(("def ", "class ", "import ", "from ", "@", "#"))
                    or line.startswith((" ", "\t"))  # indented code
                ):
                    code_lines.append(line)
                else:
                    # hit something that looks like prose → stop
                    break

        if code_lines:
            text = "\n".join(code_lines).strip()

        return text


    def generate(self, prompt: str, max_tokens: int = 512) -> str:
        # Format as chat for Qwen3
        messages = [{"role": "user", "content": prompt}]
        prompt_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

        inputs = self.tokenizer(
            prompt_text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_input_tokens,
        ).to(self.device)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=False,
            temperature=0.0,
            top_p=1.0,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        # 1) Keep only newly generated tokens (strip the prompt)
        gen_ids = outputs[0, inputs["input_ids"].shape[1]:]
        raw = self.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
        text = self._clean_generation(raw)

        # Debug (keep for now):
        #full = self.tokenizer.decode(outputs[0], skip_special_tokens=False)
        #print("=== FULL ===")
        #print(full)
        #print("=== RAW GEN ===")
        #print(raw)
        #print("=== CLEANED ===")
        #print(text)

        if not self._looks_like_python_module(text):
            print("=== REJECTED: does not look like a Python module ===")
            return ""

        return text
