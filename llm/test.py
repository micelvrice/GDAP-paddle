from paddlenlp.transformers import AutoModelForCausalLM, AutoTokenizer
import inspect
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b")

print(inspect.getfile(model.llama.layers[0].self_attn.forward))
