import matplotlib.pyplot as plt
import matplotlib.image as mpimg

img = mpimg.imread("/Users/thomasfairweather/Downloads/best_route_map_allforecast_20240616.png")

fig, ax = plt.subplots(figsize=(15, 5))
ax.imshow(img)


fig, ax = plt.subplots(figsize=(15, 5))
ax.imshow(img)
ax.set_xticks([])
ax.set_yticks([])
ax.spines[['top','bottom','left','right']].set_visible(False)

ax.set_title("North Atlantic Routes by Policy: Departure 16 June 2024",
             fontsize=13, pad=6)
ax.set_xlabel("Longitude", fontsize=11, labelpad=4)
ax.set_ylabel("Latitude", fontsize=11, labelpad=4)

plt.tight_layout(pad=0.5)
plt.savefig("route_plot_clean.png", dpi=200, bbox_inches='tight')
plt.show()