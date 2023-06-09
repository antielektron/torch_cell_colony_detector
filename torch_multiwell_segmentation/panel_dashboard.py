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

hv.extension('bokeh')

from .ml_worker import MLWorker

# Initialize the worker
ml_worker = MLWorker()

custom_css = """
.custom-file-input input[type=file] {
    display: block;
    width: 100%;
    aspect-ratio: 1;
    padding: 30px;
    cursor: pointer;
    text-align: center;
    border: 2px dashed #bbb;
    border-radius: 10px;
    background-color: #f0f0f0;
    fill: #bbb;
    content: 'Drag and Drop or Click to Upload';
    display: inline-block;
    margin: 2px;
    font-size: 16px;
}

"""

# extensions for alerts
pn.extension(notifications=True, raw_css=[custom_css])


def create_dashboard():
    
    # Initialize the template
    template = pn.template.MaterialTemplate(title="Multiwell Segmentation Dashboard")

    df = pd.DataFrame({
        'Metric': ['Ratio of masked pixels'],
        'Value': [0.12],  # Just a placeholder value, you'd update this with the actual value
    })

    # Turn the DataFrame into a Panel widget
    df_widget = pn.widgets.DataFrame(df, name='Metrics', width=500, height=200)

    def df_to_excel(df):
        output = StringIO()
        df.to_csv(output, index=False)
        output.seek(0)
        return output

    # Create a button for downloading the DataFrame as an Excel file
    download_button = FileDownload(
        callback=pn.bind(df_to_excel, df_widget.value), filename='metrics.csv')

    

    result_stream = Stream.define('Result', result=None)()
    input_stream = Stream.define('Input', input=None)()
    def get_result(result=None, input=None):
        # This function just returns the result. It will be called by DynamicMap each time the result stream is updated.
        if result is not None and input is not None:
            height, width = result[0].shape
            extents = (0, 0, width, height)
            mask1 = np.zeros((input.shape[0], input.shape[1], 4), dtype=float)
            mask2 = np.zeros((input.shape[0], input.shape[1], 4), dtype=float)

            mask1[...,3] = 0
            mask2[...,3] = 0


            mask1[result[0] > 0.8, 0] = 0.5
            mask1[result[0] > 0.8, 3] = 0.5
            mask2[result[1] > 0.5, 1] = 0.5
            mask2[result[1] > 0.5, 3] = 0.5

            return hv.RGB(input, bounds=extents) * hv.RGB(mask1, bounds=extents) * hv.RGB(mask2, bounds=extents)


        else:
            extents = (0,0, 1, 1)
        return hv.RGB(None, bounds=extents) * hv.RGB(None, bounds=extents) * hv.RGB(None, bounds=extents)
    
    # Create a DynamicMap that calls get_result when the result stream is updated
    dynamic_map = hv.DynamicMap(get_result, streams=[result_stream, input_stream]).redim.range(x=(None, None), y=(None, None))

    # set the size of the map
    dynamic_map.opts(
        width=600,
        height=600,
        framewise=True,
        xaxis=None,
        yaxis=None,
    )

    # Define the widgets

    # multiple file upload
    file_input = FileInput(accept=".png,.jpg,.jpeg,.tiff,.tif", width=300, height=300, sizing_mode="stretch_both", margin=(10, 10, 10, 10), css_classes=["custom-file-input"])
    process_button = Button(name="Process")


    # Add the widgets to the sidebar
    template.sidebar.append(file_input)
    template.sidebar.append(process_button)

    # Create an empty DynamicMap
    template.main.append(
        pn.Row(
            dynamic_map,
            df_widget,
            sizing_mode="stretch_both",
            margin=(10, 10, 10, 10),
        )
    )
    template.sidebar.append(download_button)


    # Async function to add tasks to the worker and retrieve the result
    async def process_image():
        buffer = BytesIO()
        buffer.write(file_input.value)
        buffer.seek(0)
        image = Image.open(buffer)
        image = np.array(image)
        image = image.astype(np.float32)
        image = (image[::2, ::2] + image[1::2, ::2] + image[::2, 1::2] + image[1::2, 1::2]) / 4
        image /= 255.0
        task_id = ml_worker.add_task(image)
        result_future = ml_worker.get_result_future(task_id)
        result = await asyncio.wrap_future(result_future)
        result_stream.event(result=result)
        input_stream.event(input=image)
        new_df = pd.DataFrame({
            'Metric': ['Ratio of masked pixels'],
            'Value': [np.nanmean(result[1,...])],
        })
        df_widget.value = new_df


    # When the button is clicked, schedule the process_image function to be run
    def on_button_click(event):
        pn.state.curdoc.add_next_tick_callback(process_image)

    process_button.on_click(on_button_click)

    return template
