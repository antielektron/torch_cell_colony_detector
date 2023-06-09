import panel as pn
import holoviews as hv
from queue import Queue
from panel.widgets import FileInput, Button
from holoviews.streams import Stream
import asyncio
import holoviews as hv
from io import BytesIO, StringIO
from PIL import Image
import numpy as np

import pandas as pd
from panel.widgets import FileDownload
import cv2

hv.extension('bokeh')

from .ml_worker import MLWorker

# Initialize the worker
ml_worker = MLWorker()

custom_css = """

.custom-file-input input[type=file]::before {
    content: '';
    display: block;
    position: absolute;
    height: calc(100% - 2px); /* Adjust as needed */
    background-color: #f0f0f0; /* Same as the background color */
    border: 2px dashed #bbb; /* Same as the border color */
    border-radius: 10px;
    z-index: 1;
    text-indent: -9999px;
}

.custom-file-input input[type=file] {
    position: relative;
    z-index: 0;
    display: block;
    width: 100%;
    height: 100px;
    padding: 30px;
    cursor: pointer;
    text-align: center;
    border: 2px dashed #bbb;
    border-radius: 10px;
    background-color: #f0f0f0;
    fill: #bbb;
    display: inline-block;
    margin: 2px;
    font-size: 12px;
}



input[type=file]::file-selector-button{
  visibility: hidden;
  color: transparent;
  font-size: 0px;

}

input[type=file]::-webkit-file-upload-button{
  visibility: hidden;
  color: transparent;
  font-size: 0px;
}

"""

# extensions for alerts
pn.extension(notifications=True, raw_css=[custom_css])


def create_dashboard():
    
    # Initialize the template
    template = pn.template.MaterialTemplate(title="Multiwell Segmentation Dashboard")

    df = pd.DataFrame({
        "file": [],
        "well": [],
        "ratio": []  # Just a placeholder value, you'd update this with the actual value
    })

    # set datatypes
    df['file'] = df['file'].astype('str')
    df['well'] = df['well'].astype('int')
    df['ratio'] = df['ratio'].astype('float')

    # Turn the DataFrame into a Panel widget
    df_widget = pn.widgets.DataFrame(df, name='Metrics', width_policy="auto", height_policy='max', autosize_mode='fit_columns')

    def df_to_excel(df):
        output = StringIO()
        df.to_csv(output, index=False)
        output.seek(0)
        return output

    # Create a button for downloading the DataFrame as an Excel file
    download_button = FileDownload(
        callback=lambda df_widget=df_widget : df_to_excel(df_widget.value),
        filename='metrics.csv',
        button_type='success',
        label='Download Metrics'
    )

    
    #def get_result(result=None, input=None):
    #    # This function just returns the result. It will be called by DynamicMap each time the result stream is updated.
    #    if result is not None and input is not None:
    #        height, width = result[0].shape
    #        extents = (0, 0, width, height)
    #        mask1 = np.zeros((input.shape[0], input.shape[1], 4), dtype=float)
    #        mask2 = np.zeros((input.shape[0], input.shape[1], 4), dtype=float)
    #        mask1[...,3] = 0
    #        mask2[...,3] = 0
    #
    #
    #        mask1[result[0] > 0.8, 0] = 0.5
    #        mask1[result[0] > 0.8, 3] = 0.5
    #        mask2[result[1] > 0.5, 1] = 0.5
    #        mask2[result[1] > 0.5, 3] = 0.5
    #
    #        return hv.RGB(input, bounds=extents) * hv.RGB(mask1, bounds=extents) * hv.RGB(mask2, bounds=extents)
    #
    #
    #    else:
    #        extents = (0,0, 1, 1)
    #    return hv.RGB(None, bounds=extents) * hv.RGB(None, bounds=extents) * hv.RGB(None, bounds=extents)
    
    ## Create a DynamicMap that calls get_result when the result stream is updated
    #dynamic_map = hv.DynamicMap(get_result, streams=[result_stream, input_stream]).redim.range(x=(None, None), y=(None, None))

    ## set the size of the map
    #dynamic_map.opts(
    #    width=600,
    #    height=600,
    #    framewise=True,
    #    xaxis=None,
    #    yaxis=None,
    #)

    # Define the widgets

    # multiple file upload
    file_input = FileInput(accept=".png,.jpg,.jpeg,.tiff,.tif", width=300, height=100, css_classes=["custom-file-input"], multiple=False)
    process_button = Button(name="Process")

    # create buttons to rotate/crop the image with unicode icons as names
    rotate_left_button = Button(name="\u21BA", width_polix="max")
    rotate_right_button = Button(name="\u21BB", width_polix="max")
    crop_in_button = Button(name="\u002B", width_polix="max")

    # put them in a row
    rotate_buttons = pn.Row(rotate_left_button, rotate_right_button, crop_in_button, width=300)

    def load_image(block_buttons=True):
        if block_buttons:
            rotate_buttons.disabled = True
            rotate_buttons.loading = True
        buffer = BytesIO()
        buffer.write(file_input.value)
        buffer.seek(0)
        image = Image.open(buffer)
        image = np.array(image)
        return image

    def save_image(image):
        buffer = BytesIO()
        image = Image.fromarray(image)
        image.save(buffer, format="PNG")
        buffer.seek(0)
        rotate_buttons.disabled = False
        rotate_buttons.loading = False
        return buffer.getvalue()

    # function to rotate the image in the file_input

    def rotate_image_left(event):
        image = load_image()
        image = np.rot90(image)
        file_input.value = save_image(image)

    def rotate_image_right(event):
        image = load_image()
        image = np.rot90(image, k=-1)
        file_input.value = save_image(image)

    def crop_image(event):
        image = load_image()
        # crop 10 % of the image
        min_x = int(image.shape[0] * 0.05)
        max_x = int(image.shape[0] * 0.95)
        min_y = int(image.shape[1] * 0.05)
        max_y = int(image.shape[1] * 0.95)

        # make sure values are still a multiple of 2
        if min_x % 2 != 0:
            min_x += 1
        if max_x % 2 != 0:
            max_x -= 1
        if min_y % 2 != 0:
            min_y += 1
        if max_y % 2 != 0:
            max_y -= 1

        image = image[min_x:max_x, min_y:max_y]
        file_input.value = save_image(image)
    
    # add the event handlers
    rotate_left_button.on_click(rotate_image_left)
    rotate_right_button.on_click(rotate_image_right)
    crop_in_button.on_click(crop_image)

    subplot_column = pn.GridBox(
        name="Subplots",
        ncols=2,
    )
    
    def create_subplots(image, well_mask, cell_mask):

        subplot_column.clear()

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            well_mask.astype(np.uint8)
        )

        # Get indices of sorted centroids by both x-coordinate (0) and y-coordinate (1)
        indices = np.lexsort((centroids[:, 0] // 100, centroids[:, 1] // 100))

        # Use indices to sort stats and centroids
        stats = stats[indices]
        centroids = centroids[indices]

        # Create a mapping from old labels to new ones
        mapping = {old: new for new, old in enumerate(indices)}

        # Create a new labels array with sorted labels
        sorted_labels = np.copy(labels)
        for old, new in mapping.items():
            sorted_labels[labels == old] = new

        labels = sorted_labels

        # only keep well labels that have at least one pixel in the cell mask and are not the background
        well_labels = np.unique(labels[cell_mask & (labels != 0)])


        # check again and through out wells that are smaller or greater than 70 % of the median well size
        well_labels = well_labels[
            (stats[well_labels, cv2.CC_STAT_AREA] > np.median(stats[well_labels, cv2.CC_STAT_AREA]) * 0.7)
            & (stats[well_labels, cv2.CC_STAT_AREA] < np.median(stats[well_labels, cv2.CC_STAT_AREA]) * 1.3)
        ]

        # create a new mask with only the well labels
        well_mask = np.isin(labels, well_labels)

        # remove the background label
        well_mask[labels == 0] = False

        for i, well in enumerate(np.unique(labels[well_mask])):
            # get the bounding box for the well
            x, y, w, h, area = stats[well]
            
            # add a small buffer
            buffer = 20
            x -= buffer
            y -= buffer
            w += buffer * 2
            h += buffer * 2


            if x < 0:
                x = 0
            if y < 0:
                y = 0
            if x + w > image.shape[1]:
                w = image.shape[1] - x
            if y + h > image.shape[0]:
                h = image.shape[0] - y
            

            # crop the well mask and cell mask
            well_mask_crop = well_mask[y : y + h, x : x + w]
            cell_mask_crop = cell_mask[y : y + h, x : x + w]
            
            # create a new image with the well mask overlay
            img_crop = image[y : y + h, x : x + w].copy()

            well_mask_crop_rgba = np.zeros((img_crop.shape[0], img_crop.shape[1], 4), dtype=float)
            cell_mask_crop_rgba = np.zeros((img_crop.shape[0], img_crop.shape[1], 4), dtype=float)
            well_mask_crop_rgba[...,3] = 0
            cell_mask_crop_rgba[...,3] = 0

            well_mask_crop_rgba[well_mask_crop > 0.5, 0] = 1
            well_mask_crop_rgba[well_mask_crop > 0.5, 3] = 0.3
            cell_mask_crop_rgba[cell_mask_crop > 0.5, 1] = 1
            cell_mask_crop_rgba[cell_mask_crop > 0.5, 3] = 0.3

            extents = (0, 0, img_crop.shape[1], img_crop.shape[0])

            # overlay the well mask on the image, giving the combined it's own dimension
            img_crop = hv.RGB(img_crop, bounds=extents) * hv.RGB(well_mask_crop_rgba, bounds=extents) * hv.RGB(cell_mask_crop_rgba, bounds=extents)

            card = pn.Card(
                img_crop.opts(width=300, height=300, framewise=True, xaxis=None, yaxis=None),
                title=f"Well {i+1}",
                sizing_mode="stretch_both",
                margin=(10, 10, 10, 10),
            )

            # add a ok and discard button
            
            ok = pn.widgets.Button(name="Ok", button_type="primary", width=100)
            discard = pn.widgets.Button(name="Discard", button_type="danger", width=100)
          
            card.append(
                pn.Row(
                    ok,
                    discard,
                )
            )

            # on ok, the mean pixel value of the well is calculated and the well is added to the well dataframe
            

            well_mean = np.sum(cell_mask_crop > 0.5).astype(np.float32) / np.sum((well_mask_crop > 0.5).astype(np.float32))
            
            def on_ok(event, i=i, fname=file_input.filename, well_mean=well_mean,df_widget=df_widget, ok=ok, discard=discard):

                well_df = df_widget.value
                
                well_df = well_df.append(
                    {
                        "file": str(fname),
                        "well": int(i+1),
                        "ratio": well_mean,
                    },
                    ignore_index=True,
                )
                df_widget.value = well_df
                ok.disabled = True
                discard.disabled = True

            
            ok.on_click(on_ok)

            # on discard, the card is removed from the layout
            def on_discard(event, card=card):
                subplot_column.remove(card)

            discard.on_click(on_discard)

            subplot_column.append(
                card
            )


    # Async function to add tasks to the worker and retrieve the result
    async def process_image():

        image = load_image(block_buttons=False)
        image = image.astype(np.float32)
        image = (image[::2, ::2] + image[1::2, ::2] + image[::2, 1::2] + image[1::2, 1::2]) / 4
        image /= 255.0
        task_id = ml_worker.add_task(image)
        result_future = ml_worker.get_result_future(task_id)
        prediction = await asyncio.wrap_future(result_future)

        well_mask = prediction[0] > 0.5
        cell_mask = prediction[1] > 0.5

        create_subplots(image, well_mask, cell_mask)

        # enable the process button

        process_button.disabled = False
        process_button.loading = False


    template.main.append(
        pn.Row(
            subplot_column,
            df_widget,
            sizing_mode="stretch_both",
            margin=(10, 10, 10, 10),
        )
    )

    # image preview
    @pn.depends(file_content = file_input.param.value)
    def image_preview(file_content):
        if file_content is None:
            return pn.pane.PNG(object=None, width=0, height=0)
        else:
            return pn.pane.PNG(object=file_content, width=300)
    

    # Add the widgets to the sidebar
    template.sidebar.append(pn.pane.Markdown("### Upload an image"))
    template.sidebar.append(pn.pane.Markdown("drag and drop an image below:"))
    template.sidebar.append(file_input)


    # Add the process button
    template.sidebar.append(pn.pane.Markdown("### Process the image"))
    template.sidebar.append(process_button)

    template.sidebar.append(pn.pane.Markdown("### Download metrics"))
    template.sidebar.append(download_button)

    template.sidebar.append(pn.pane.Markdown("### Image preview"))
    template.sidebar.append(rotate_buttons)
    template.sidebar.append(image_preview)


    # When the button is clicked, schedule the process_image function to be run
    def on_button_click(event):
        # check if a file has been uploaded
        if file_input.filename is None:
            pn.state.notifications.error("Please upload an image first", duration=10000)
            return

        # disable the process button
        process_button.disabled = True
        process_button.loading = True

        subplot_column.clear()
        subplot_column.append(pn.indicators.LoadingSpinner(value=True, width=600, height=600))

        # schedule one time callback to process the image
        pn.state.curdoc.add_next_tick_callback(process_image)

    process_button.on_click(on_button_click)

    return template
