import torch
from threading import Thread, Event
from queue import Queue
from uuid import uuid4
from concurrent.futures import Future
import numpy as np
from pathlib import Path
from torch_multiwell_segmentation.tiled_prediction import TiledPrediction
import logging

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class MLWorker:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(MLWorker, cls).__new__(cls, *args, **kwargs)
        return cls._instance

    def __init__(self):
        self.task_queue = Queue()
        self.model = self.load_model()
        self.results = {}
        self.stop_signal = Event()  # Add a stop signal
        # Start the worker thread
        self.worker_thread = Thread(target=self.worker)
        self.worker_thread.start()

    def load_model(self):
        tiled_prediction = TiledPrediction(
            tile_size=(256, 256),
            model_path=str(Path(__file__).parent / 'data' / 'model.pt'),
            overlap=(12, 12),
            device=device,
            input_channels=3,
            output_channels=2,
        )
        return tiled_prediction


    def process_image(self, image):
        # Process image with the model
        array = image.astype(np.float32)

        return self.model(array)
        

    def worker(self):
        logging.info('Worker started')
        while not self.stop_signal.is_set():
            # Get a new task from the queue

            try:
                task = self.task_queue.get(timeout=1)
            except:
                if self.stop_signal.is_set():
                    break
                continue
            # Process the image
            logging.info('Processing image')
            result = self.process_image(task['image'])

            # Set the result in the Future
            self.results[task['id']].set_result(result)
        
        logging.info('Worker stopped')

    def add_task(self, image):
        task_id = uuid4().hex
        task_future = Future()
        self.task_queue.put({'id': task_id, 'image': image})
        self.results[task_id] = task_future
        return task_id

    def get_result_future(self, task_id):
        # Return the Future for the task
        return self.results[task_id]

    def stop(self):  # Add a stop method
        self.stop_signal.set()
        self.task_queue.put(None)  # Put a None task to unblock the queue
        self.worker_thread.join()  # Wait for the worker thread to finish

