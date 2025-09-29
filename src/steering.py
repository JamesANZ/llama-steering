import torch
from nnsight import LanguageModel
import psutil
from huggingface_hub import hf_hub_download
from huggingface_hub import InferenceClient
import os

BILL_LLM_EVALUATION_TO = "<...>"     # Replace with your billing account ID on Hugging Face Inference Providers
LLM_FOR_EVALUATION = "openai/gpt-oss-120b"

def load_saes(cfg, device):
    """ Load steering vectors from SAEs and prepare steering components."""
    if not cfg['features'] or len(cfg['features']) == 0:
        print("No features specified, returning empty steering components.")
        return []

    steering_components = []
    cache_dir = "./downloads"
    features = cfg['features']
    reduced_strengths = cfg.get('reduced_strengths', False)

    for i in range(len(features)):
        layer_idx, feature_idx = features[i][0], features[i][1]
        strength = features[i][2] if len(features[i]) > 2 else 0.0

        # If the strengths in the config file were given in reduced form, scale them by layer index
        if reduced_strengths:
            strength *= layer_idx

        print(f"Loading feature {layer_idx} {feature_idx} {strength:.2f} [{strength/layer_idx:.2f}]")

        sae_filename = cfg['sae_filename_prefix'] + f"{layer_idx}" + cfg['sae_filename_suffix']
        file_path = hf_hub_download(repo_id=cfg['sae_path'], filename=sae_filename, cache_dir=cache_dir)
        sae = torch.load(file_path, map_location="cpu")
        vec = sae["decoder.weight"][:, feature_idx].to(device, non_blocking=True)
        steering_components.append({'layer': layer_idx, 'feature': feature_idx, 'strength': strength, 'vector': vec / vec.norm()})
        del sae

    return steering_components


def rep_n(token_ids, n):
    """ Compute rep-n metric for a sequence of token IDs. rep-n = 1 - (unique n-grams / total n-grams) """
    if len(token_ids) < n:
        return 0.0
    total = len(token_ids) - n + 1
    unique = len({tuple(token_ids[i:i + n]) for i in range(total)})
    return 1 - (unique / total)


def print_memory_usage():
    """ Prints current RAM and GPU memory usage."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mem = psutil.virtual_memory()
    used_gb = mem.used / 1e9
    total_gb = mem.total / 1e9
    available_gb = mem.available / 1e9
    percent = mem.percent
    gpu_mem = (torch.cuda.memory_allocated() / 1e9) if device == "cuda" else 0.0
    print(f"GPU mem (GB): {gpu_mem:.1f} --- ", end="")
    print(f"RAM: {used_gb:.1f}/{total_gb:.1f} GB used ({percent:.1f}%), {available_gb:.1f} GB available")

    proc = psutil.Process(os.getpid())

    rss_bytes = proc.memory_info().rss               # resident set size
    rss_gb = rss_bytes / (1024 ** 3)

    # include children (useful if you spawn workers / subprocess LLM calls)
    child_rss_bytes = sum(c.memory_info().rss for c in proc.children(recursive=True))
    total_rss_gb = (rss_bytes + child_rss_bytes) / (1024 ** 3)

    gpu_gb = torch.cuda.memory_allocated() / (1024 ** 3) if torch.cuda.is_available() else 0.0

    print(f"GPU={gpu_gb:.1f} GB | RSS self={rss_gb:.1f} GB | RSS self+children={total_rss_gb:.1f} GB")


def generate_steered_answer(llm : LanguageModel,
                            chat,
                            steering_components,
                            max_new_tokens=128,
                            temperature = 0.0,
                            repetition_penalty = 1.0,
                            steer_prompt = True,
                            clamp_intensity = False):
    """
    Generates an answer from the model given a chat history, applying steering components.
    Expects steering_components to be a list of dicts with keys:
        'layer': int, layer index to apply steering
        'strength': float, steering intensity
        'vector': torch.Tensor, steering vector
    """
    input_ids = llm.tokenizer.apply_chat_template(chat, tokenize=True, add_generation_prompt=True)
    with llm.generate(max_new_tokens=max_new_tokens, repetition_penalty = repetition_penalty,
                      do_sample=temperature > 0.0, temperature=temperature,
                      pad_token_id=llm.tokenizer.eos_token_id) as tracer:
        with tracer.invoke(input_ids):
            with tracer.all() as idx:
                for sc in steering_components:
                    layer, strength, vector = sc["layer"], sc["strength"], sc["vector"]
                    length = llm.model.layers[layer].output.shape[1]
                    amount = (strength * vector).unsqueeze(0).expand(length, -1).unsqueeze(0).clone()
                    if clamp_intensity:
                        projection = (llm.model.layers[layer].output @ vector).unsqueeze(-1)@(vector.unsqueeze(0))
                        amount -= projection
                    llm.model.layers[layer].output += (steer_prompt or idx > 0) * amount
        with tracer.invoke():
            trace = llm.generator.output.save()

    answer = llm.tokenizer.decode(trace[0][len(input_ids):], skip_special_tokens=True)
    output = {'input_ids': input_ids, 'trace': trace, 'answer': answer}
    return output


def compute_metrics(llm, output, use_llm_evaluation = False, instruction = None, concept = None):

    trace = output['trace']
    input_ids = output['input_ids']

    with torch.inference_mode():
        out = llm(input_ids=trace, use_cache=False)
        logits = out.logits.detach().to("cpu")
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

    avg_log_prob : float = 0.0
    greedy_avg_log_prob : float = 0.0
    for i in range(len(input_ids), len(trace[0]) - 1):

        # log_prob of the chosen token in the original model
        token_id = trace[0, i].item()
        avg_log_prob += log_probs[0, i - 1, token_id].item()

        # log_prob of the greedy token in the original model
        greedy_token_id = torch.argmax(log_probs[0, i - 1, :]).item()
        greedy_avg_log_prob += log_probs[0, i - 1, greedy_token_id].item()

    avg_log_prob /= (len(trace[0]) - len(input_ids) - 1)
    greedy_avg_log_prob /= (len(trace[0]) - len(input_ids) - 1)

    answer_tokens = output['trace'][0][len(output['input_ids']):].clone().tolist()
    rep3, rep4 = rep_n(answer_tokens, 3), rep_n(answer_tokens, 4)

    result = {'delta_log_prob' : avg_log_prob - greedy_avg_log_prob, 'avg_log_prob' : avg_log_prob, 'greedy_avg_log_prob' : greedy_avg_log_prob, 'rep3' : rep3, 'rep4' : rep4}

    assert instruction is not None and concept is not None if use_llm_evaluation else True
    if use_llm_evaluation:
        llm_results = llm_evaluation(instruction, output['answer'], concept, verbose = False)
        result.update(llm_results)

    return result


# ======================================================================================================================

# ----------------------
#  LLM-based evaluation
# ----------------------

prompt_concept = """[System]
Please act as an impartial judge and evaluate whether the specified concept is presented in the text fragment provided below.
Focus solely on whether the concept is clearly incorporated, without regard for grammar or logical coherence.
Rate the concept’s relevance on a scale from 0 to 2, where 0 indicates the concept is not present at all, 1 indicates the concept is somewhat
present but minimally or awkwardly incorporated, and 2 indicates the concept is more fully and effectively incorporated, with
stronger and more natural integration. Provide your rating using this exact format: “Rating: [[score]]”.

[Concept Start]
{concept}
[Concept End]

[Text Fragment Start]
{answer}
[Text Fragment End]"""

prompt_instruction = """
Please act as an impartial judge and evaluate whether the text fragment provided below is related to the instruction. Focus
solely on the degree of relatedness in terms of topic, regardless of grammar, coherence, or informativeness.
Rate the instruction relevance on a scale from 0 to 2,
where 0 indicates the text is unrelated to the instruction, 1 indicates it is somewhat related but only minimally or indirectly
relevant in terms of topic, and 2 indicates it is more clearly and directly related to the instruction. Provide your rating using this
exact format: “Rating: [[score]]”.

[Instruction Start]
{instruction}
[Instruction End]

[Text Fragment Start]
{answer}
[Text Fragment End]
"""

prompt_fluency = """
Please act as an impartial judge and evaluate the fluency of the text fragment provided below. Focus solely on fluency,
disregarding its completeness, relevance, coherence with any broader context, or informativeness.
In particular, since this is a text fragment, do not penalize it for being incomplete or lacking context.
Consider the fluency of the text, noting any unnatural phrasing, awkward transitions,
grammatical errors, or repetitive structures that may hinder readability. Rate the text’s
fluency on a scale from 0 to 2, where 0 indicates the text is not fluent and highly unnatural (e.g., incomprehensible or repetitive),
1 indicates it is somewhat fluent but contains noticeable errors or awkward phrasing, and 2 indicates the text is fluent and
almost perfect. Provide your rating using this exact format: “Rating: [[score]]”.

[Text Fragment Start]
{answer}
[Text Fragment End]
"""

def extract_rating(response):
    for i in [0, 1, 2]:
        if str(i) in response:
            return i
    return None

def llm_evaluation(instruction, answer, concept, verbose = False):

    client = InferenceClient(bill_to=BILL_LLM_EVALUATION_TO)
    model = LLM_FOR_EVALUATION

    def safe_llm_call(prompt_text, prompt_name):
        """Safely call the LLM and return rating or None on error."""
        try:
            completion = client.chat.completions.create(model=model,
                messages=[{"role": "user", "content": prompt_text}],)
            rating = extract_rating(completion.choices[0].message.content)
            if verbose:
                print(f"{prompt_name} rating:", rating, "Response:", completion.choices[0].message.reasoning)
            return rating
        except Exception as e:
            if verbose:
                print(f"{prompt_name} evaluation failed: {e}.")
            return 0.0

    rating_concept = safe_llm_call(prompt_concept.format(concept=concept, answer=answer),"Concept")
    rating_instruction = safe_llm_call(prompt_instruction.format(instruction=instruction, answer=answer),"Instruction")
    rating_fluency = safe_llm_call(prompt_fluency.format(answer=answer),"Fluency")

    return {'concept': rating_concept, 'instruction': rating_instruction, 'fluency': rating_fluency}
