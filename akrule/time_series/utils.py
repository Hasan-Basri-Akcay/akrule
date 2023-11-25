import matplotlib.pyplot as plt

def plot_pred(X_val=None, X_pred=None, tag_features=None, figsize=(16,4)):
    fig, ax = plt.subplots(figsize=figsize)
    colors = ["tab:blue", "tab:red"]
    for index, (group_name, group) in enumerate(X_val.groupby(tag_features)):
        ax.plot(group.time, group.y, color=colors[index], label=group_name)
        ax.fill_between(group.time, group.y_upper, group.y_lower, facecolor=colors[index], alpha=0.5) 
    ax.axvline(x=X_pred.time.min(), ls='--', color="black", alpha=0.5)
    for index, (group_name, group) in enumerate(X_pred.groupby(tag_features)):
        ax.plot(group.time, group.y_pred, color=colors[index], label=group_name)
        ax.fill_between(group.time, group.y_upper, group.y_lower, facecolor=colors[index], alpha=0.5) 
    ax.legend(loc='upper left');