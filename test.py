from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

# Initialize the tokenizer
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")

# Configurae the sampling parameters (for thinking mode)
sampling_params = SamplingParams(temperature=0.7, top_p=0.8, top_k=20, max_tokens=32768, n = 64)

# Initialize the vLLM engine
llm = LLM(model="Qwen/Qwen3-0.6B")

# Prepare the input to the model
# prompt = "Your task is to translate between English and the low resource language of Central Aymara. Here are some examples:\n\nExample 1:\n\nEnglish: We got up a five a.m. and took a 3 mile jaunt all around our neighborhood. It was great exercise and a great way to start the day! How did your day start today?\Central Aymara: Phisqha alwa pachaw sartapxta ukatx utaj jak’an 3 millas ukaruw muytir sarapxta. Jichhürux kusapuniw ukham muytir sarañaxa! Jumatakist kunjamakis jichhüruxa?\n\nExample 2:\n\nEnglish: It would be awesome to go to one of your shows! Do you have anything coming up?\Central Aymara: Walikipuniw jutawa! Jutir urutak utjtamti?\n\nNow your turn. Translate the following English sentence into Aymara:\n\nEnglish: What's the weather like in Arequipa today?\Central Aymara:"
prompt = "Hello. How are you?"
messages = [
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=False,  # Set to False to strictly disable thinking
)

# Generate outputs
outputs = llm.generate([text], sampling_params)

# Print the outputs.
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"{generated_text!r}")
