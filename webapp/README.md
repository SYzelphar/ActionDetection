# Action Detection UI

A browser frontend for `action.keras`, the sign-language LSTM from the notebook. Every frame goes through MediaPipe Holistic to get 1,662 keypoints (pose, face and both hands). The model then classifies the last 30 frames as one of the actions in `labels.json`.

## Run

Double-click `run_ui.bat` in the project root, or run:

```
.venv\Scripts\python -m uvicorn webapp.server:app --port 8000
```

Then open http://localhost:8000. The first time you run `run_ui.bat`, it creates `.venv` with Python 3.11, TensorFlow 2.18 (Keras 3, which the model was saved with) and MediaPipe 0.10.21.

## Features

- **Live camera:** frames stream to the server over a WebSocket. Pose, hand and face landmarks are drawn on the video. The recognized action shows as a caption, with probability bars for every class.
- **Video file:** drop in a recorded clip and it's analysed while it plays. Seeking or looping restarts the 30-frame buffer.
- **Recognized:** a running list of detected actions, like the `sentence` in the notebook.
- **Stable prediction:** on by default, as in the refined notebook. An action only counts once it has been the top guess for 10 frames in a row and is above the confidence threshold.

## Changing the actions

The class names live in `webapp/labels.json`, in the same order they were trained. If you retrain with different actions, update that file. The server refuses to start if the number of labels doesn't match the model's output size.

## API

- `WS /ws` takes a JPEG frame as a binary message and replies with JSON: `{landmarks, buffered, probs, holistic_ms, model_ms}`. `probs` is `null` until 30 frames have been buffered. Send the text message `reset` to clear the buffer.
- `GET /api/info` returns the labels, sequence length and landmark connections.
