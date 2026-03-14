import asyncio
import argparse
from loguru import logger
from core.pipeline import AsyncVisionPipeline

async def on_detection_callback(frame, results):
    \"\"\"
    Sample callback function to handle detections.
    \"\"\"
    # results[0].boxes contains detected objects
    num_detections = len(results[0].boxes)
    if num_detections > 0:
        logger.info(f"Compliance Check: Detected {num_detections} safety elements.")

async def main():
    parser = argparse.ArgumentParser(description="VisionSpectra Safety Monitor CLI")
    parser.add_argument("--source", type=str, default="0", help="Camera index or video file path")
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="Path to YOLO weights")
    args = parser.parse_args()

    logger.add("logs/vision_spectra.log", rotation="10 MB")
    logger.info("Starting VisionSpectra...")

    pipeline = AsyncVisionPipeline(model_path=args.model)
    
    try:
        await pipeline.run(source=args.source, callback_func=on_detection_callback)
    except KeyboardInterrupt:
        logger.info("Shutdown initiated by user.")
    except Exception as e:
        logger.exception(f"Unexpected error: {str(e)}")
    finally:
        logger.info("VisionSpectra halted.")

if __name__ == "__main__":
    asyncio.run(main())