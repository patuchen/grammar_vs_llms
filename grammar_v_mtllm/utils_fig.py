import matplotlib as mpl
import matplotlib.pyplot as plt


COLORS = [
    "#bc272d",  # red
    "#50ad9f",  # green
    "#0000a2",  # blue
    "#954ead",  # purple
    "#e9c716",  # yellow
    "#e67e22",  # orange
]

LANG_TO_COLOR = {
    "cs-uk": "#0057B7",
    "en-de": "#000000",
    "en-zh": "#B22222",
}
LANG_TO_NAME = {
    "cs-uk": r"Czech$\rightarrow$Ukrainian",
    "en-de": r"English$\rightarrow$German",
    "en-zh": r"English$\rightarrow$Chinese",
}

mpl.rcParams["font.family"] = "serif"
mpl.rcParams["axes.prop_cycle"] = plt.cycler(color=COLORS)
mpl.rcParams["legend.fancybox"] = False
mpl.rcParams["legend.edgecolor"] = "None"
mpl.rcParams["legend.fontsize"] = 9
mpl.rcParams["legend.borderpad"] = 0.1



def turn_off_spines(which=['top', 'right'], ax=None):
    import matplotlib.pyplot as plt

    if ax is None:
        ax = plt.gca()
    ax.spines[which].set_visible(False)
