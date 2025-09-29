import torch
from nnsight import LanguageModel
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
llm = LanguageModel("meta-llama/Llama-3.1-8B-Instruct", dispatch=True).to(device)

def compute_residual_norms(llm, input_ids):
    layer_norms = []

    with llm.trace(input_ids):
        for layer_idx, layer in enumerate(llm.model.layers):
            residual = layer.output
            norms = torch.norm(residual, dim=-1).save()  # (batch_size, seq_len)
            layer_norms.append(norms)

    return layer_norms


prompt = """
Call me Ishmael. Some years ago—never mind how long precisely—having little or no money in my purse, and nothing particular to interest me on shore, I thought I would sail about a little and see the watery part of the world. It is a way I have of driving off the spleen and regulating the circulation. Whenever I find myself growing grim about the mouth; whenever it is a damp, drizzly November in my soul; whenever I find myself involuntarily pausing before coffin warehouses, and bringing up the rear of every funeral I meet; and especially whenever my hypos get such an upper hand of me, that it requires a strong moral principle to prevent me from deliberately stepping into the street, and methodically knocking people’s hats off—then, I account it high time tozz get to sea as soon as I can. This is my substitute for pistol and ball. With a philosophical flourish Cato throws himself upon his sword; I quietly take to the ship. There is nothing surprising in this. If they but knew it, almost all men in their degree, some time or other, cherish very nearly the same feelings towards the ocean with me.
There now is your insular city of the Manhattoes, belted round by wharves as Indian isles by coral reefs—commerce surrounds it with her surf. Right and left, the streets take you waterward. Its extreme downtown is the battery, where that noble mole is washed by waves, and cooled by breezes, which a few hours previous were out of sight of land. Look at the crowds of water-gazers there.
Circumambulate the city of a dreamy Sabbath afternoon. Go from Corlears Hook to Coenties Slip, and from thence, by Whitehall, northward. What do you see?—Posted like silent sentinels all around the town, stand thousands upon thousands of mortal men fixed in ocean reveries. Some leaning against the spiles; some seated upon the pier-heads; some looking over the bulwarks of ships from China; some high aloft in the rigging, as if striving to get a still better seaward peep. But these are all landsmen; of week days pent up in lath and plaster— tied to counters, nailed to benches, clinched to desks. How then is this? Are the green fields gone? What do they here?
"""

input_ids = llm.tokenizer.encode(prompt, return_tensors="pt")
norms = compute_residual_norms(llm, input_ids)

# Convert norms to numpy
norms = [n.cpu().detach().numpy() for n in norms]

# Save norms to a csv file
import numpy as np
np.savetxt("residual_norms.csv", np.array(norms).squeeze(), delimiter=",")

# Compute means
means = []
for l in range(len(norms)):
    mean = norms[l][0][1:].mean()
    means.append(mean)

# Save means
np.savetxt("residual_means.csv", np.array(means), delimiter=",")

from matplotlib import pyplot as plt
fig = plt.figure(figsize = (13,5))
fig.suptitle("Residual stream activations magnitude across layers and tokens")
plt.subplot(1,2,1)
means = []
for l in range(len(norms)):
    plt.plot(norms[l][0].cpu().detach().numpy(), label=f"Layer {l}")
plt.xlim(0, input_ids.shape[1])
plt.ylim(0,60)
plt.xlabel("Token index")
plt.ylabel("Norm of residual stream")

plt.subplot(1,2,2)
plt.plot(means,'o-')
plt.xlabel("Layer index")
plt.ylabel("Norm of residual stream")
plt.xlim(0, len(norms))
plt.ylim(0,60)
plt.savefig("figures/activations_magnitude.png", dpi=300, bbox_inches='tight')


