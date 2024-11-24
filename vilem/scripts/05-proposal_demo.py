import matplotlib.pyplot as plt

plt.figure(figsize=(4, 2.5))

COLORS = [
    "darkslateblue",
    "darkgoldenrod",
    "seagreen",
]

gpt_4o_mini = [
    (70, 0.96),
    (68, 0.98),
    (66, 0.99),
    (60, 0.98),
    (53, 0.92),
    (40, 0.83),
]

seth_v1_eurollm = [
    (50, 0.95),
    (37, 0.73),
    (27, 0.51),
    (17, 0.27),
    (11, 0.17),
    (6, 0.11),
]

seth_v1_tower = [
    (50, 0.97),
    (48, 0.97),
    (40, 0.88),
    (33, 0.76),
    (30, 0.65),
    (28, 0.56),
]


plt.plot(
    [x[0] for x in gpt_4o_mini],
    label="GPT-4o mini",
    color=COLORS[0],
)
plt.plot(
    [x[0] for x in seth_v1_tower],
    label="TowerLLM",
    color=COLORS[1],
)
plt.plot(
    [x[0] for x in seth_v1_eurollm],
    label="EuroLLM",
    color=COLORS[2],
)
plt.text(
    5, gpt_4o_mini[-1][0],
    s="GPT-4o mini",
    color=COLORS[0],
    va="center",
)
plt.text(
    5, seth_v1_tower[-1][0],
    s="TowerLLM",
    color=COLORS[1],
    va="center",
)
plt.text(
    5, seth_v1_eurollm[-1][0],
    s="EuroLLM",
    color=COLORS[2],
    va="center",
)

for i, v in list(enumerate(gpt_4o_mini))[::2]:
    plt.text(
        i, v[0]+2,
        f"{v[1]:.0%}",
        ha="center",
    )
for i, v in list(enumerate(seth_v1_tower))[::2]:
    plt.text(
        i, v[0]+2,
        f"{v[1]:.0%}",
        ha="center",
    )
for i, v in list(enumerate(seth_v1_eurollm))[2::2]:
    plt.text(
        i, v[0]+2,
        f"{v[1]:.0%}",
        ha="center",
    )

plt.xlabel("Noise level")
plt.ylabel("chrF (not percentages)")
# turn of spines
plt.gca().spines[["top", "right"]].set_visible(False)

plt.annotate(
    "% of outputs in\ntarget language",
    xy=(2.4, 45),
    xytext=(5, 70),
    color="#444",
    ha="center", va="center",
    arrowprops=dict(color='#666', width=1),
)


plt.xticks(
    [0, 5],
    ["0%", "100%"],
)

plt.tight_layout(pad=0.1)
plt.savefig("/home/vilda/Downloads/proposal_demo.pdf")
plt.show()