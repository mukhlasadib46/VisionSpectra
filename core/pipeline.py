import asyncio
import cv2
import numpy as np
from loguru import logger
from typing import Optional, List, Tuple
from ultralytics import YOLO

class AsyncVisionPipeline:
    \"\"\"
    High-performance asynchronous vision pipeline for real-time inference.
    Designed to decouple frame capture from model processing to eliminate lag.
    \"\"\"
    def __init__(self, model_path: str = "yolov8n.pt"):
        self.model = YOLO(model_path)
        self.frame_queue = asyncio.Queue(maxsize=30)
        self.is_running = False
        logger.info(f"Initialized AsyncVisionPipeline with {model_path}")

    async def frame_producer(self, source: str):
        \"\"\"
        Reads frames from the source and pushes them into a non-blocking queue.
        \"\"\"
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            logger.error(f"Could not open source: {source}")
            return

        self.is_running = True
        logger.info(f"Producer started for source: {source}")

        while self.is_running:
            ret, frame = cap.read()
            if not ret:
                logger.warning("End of stream or read error.")
                break
            
            # Use non-blocking put_nowait to avoid delaying frame capture
            try:
                if self.frame_queue.full():
                    self.frame_queue.get_nowait() # Drop oldest frame
                self.frame_queue.put_nowait(frame)
            except asyncio.QueueEmpty:
                pass
            
            await asyncio.sleep(0.01) # Small sleep to yield control
        
        cap.release()
        self.is_running = False

    async def inference_consumer(self, callback_func=None):
        \"\"\"
        Consumes frames from the queue and performs inference.
        \"\"\"
        logger.info("Inference consumer started.")
        while self.is_running or not self.frame_queue.empty():
            if self.frame_queue.empty():
                await asyncio.sleep(0.05)
                continue

            frame = await self.frame_queue.get()
            
            # Perform inference (simulating real-time safety checks)
            results = self.model.predict(frame, conf=0.5, verbose=False)
            
            # Logic for safety compliance can be added here (e.g., PPE checks)
            if callback_func:
                await callback_func(frame, results)
            
            self.frame_queue.task_done()

    async def run(self, source: str, callback_func=None):
        \"\"\"
        Main entry point to start the async pipeline.
        \"\"\"
        await asyncio.gather(
            self.frame_producer(source),
            self.inference_consumer(callback_func)
        )