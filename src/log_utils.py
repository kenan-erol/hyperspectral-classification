import matplotlib.pyplot as plt
import numpy as np

def hsi_to_rgb_display(hsi_image):
    # Helper to convert HSI to displayable RGB (same as in datasets.py)
    if hsi_image.ndim == 2: hsi_image = np.expand_dims(hsi_image, axis=-1)
    img_h, img_w, img_c = hsi_image.shape
    if img_c == 0: return np.zeros((img_h, img_w, 3), dtype=np.uint8)
    display_img = np.mean(hsi_image, axis=2)
    min_val, max_val = np.min(display_img), np.max(display_img)
    if max_val > min_val:
        display_img = ((display_img - min_val) / (max_val - min_val) * 255).astype(np.uint8)
    else:
        display_img = np.zeros((img_h, img_w), dtype=np.uint8)
    return np.stack([display_img]*3, axis=-1)

def config_plot():
    '''
    Function to remove axis tickers and box around figure
    '''

    plt.box(False)
    plt.axis('off')

def plot_images(images, n_row, n_col, subplot_titles, output_path=None, dpi=200, cmap=None):
    '''
    Plot images in a grid

    Arg(s):
        images : list[list[numpy]]
            lists of lists of images
        n_row : int
            number of rows in plot
        n_col : int
            number of columns in plot
        subplot_titles : list[list[str]]
            lists of lists of titles corresponding to each subplot
        dpi : int
            dots per inch for figure
        cmap : matplotlib.Colormap
            dots per inch for figure
    '''

    # Instantiate a figure
    fig = plt.figure(dpi=dpi)

    # Iterate through each row of images
    for row_idx in range(n_row):

        # Iterate through each column of row
        for col_idx in range(n_col):

            # TODO: Compute subplot index based on row and column indices
            subplot_idx = row_idx * n_col + col_idx + 1

            # TODO: Create axis object with n_row, n_col for current subplot
            ax = fig.add_subplot(n_row, n_col, subplot_idx)

            # TODO: Plot the image with provided color
            ax.set_title(subplot_titles[row_idx][col_idx], fontsize=5)
            ax.imshow(images[row_idx][col_idx], cmap=cmap)

            config_plot()

    if output_path is not None:
        fig.subplots_adjust(wspace=0, hspace=0.5)
        plt.savefig(output_path)
    else:
        return fig
