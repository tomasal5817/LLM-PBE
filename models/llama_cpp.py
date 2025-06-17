import time
import re
import subprocess
from models.LLMBase import LLMBase

class Llama_cpp(LLMBase):
    def __init__(self,api_key=None, model=None, max_attempts=3, llama_cpp_path=None, model_path=None):
        super().__init__(api_key=api_key, model_path=model_path)
        
        self.model=model
        self.max_attempts=max_attempts
        self.delay_seconds=3
        self.tokenizer=None
        self.counter = 0
        self.llama_cpp_path = llama_cpp_path

    def load_model(self):
        pass
        
    def query_local_model(self, query, messages=None):
        pass
    
    def evaluate(self, text, tokenized=False):
        """
        Use llama.cpp to measure the perplexity of a GGUF model
        
        Parameters:
        - text (str): The text prompt to query the model.

        Returns:
        - ppl_value: The model's perplexity.
        """
        # NOTE: Functionality to

        import re
        import subprocess
        # Call llama.cpp from here and return PPL

        # llama.cpp perplexety input file
        with open("file.txt", "w") as f:
            f.write(text)
        
        model_path = self.model
        llama_cpp_path = self.llama_cpp_path
        run_llama_cpp = f'{llama_cpp_path} -m {model_path} -f file.txt -c 32'
        missing_perplexity_value = -1 # If llama.cpp is unable to get perplexity, negative perplexity will be discarded

        result = subprocess.run(run_llama_cpp, shell=True, capture_output=True, text=True)

        if result.returncode != 0:
            print("Command failed:")
            print(result.stderr)
            return missing_perplexity_value
    
        match = re.search(r"Final estimate: PPL = ([\d.]+) \+/- ([\d.]+)", result.stderr)

        if match:
            ppl_value = float(match.group(1))
            ppl_error = float(match.group(2))
            print(f"PPL: {ppl_value}, Error: {ppl_error}")
            return ppl_value
        else:
            print("PPL not found in output: discard this test value")
            return missing_perplexity_value
        
    def evaluate_ppl(self, text, tokenized=False):
        """
        Evaluate an open-source model with a given text prompt.
        
        Parameters:
        - text (str): The text prompt to query the model.

        Returns:
        - ppl_value: The model's perpelexity.
        """
        ppl_value = self.evaluate(text, tokenized=tokenized)
        return ppl_value

