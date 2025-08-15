## Assumptions

The analysis operates on a set of key assumptions derived from the project scope:

- **Consistent Point of View:** The model is tuned for a side-on camera angle, similar to the one in the provided source video. Its performance may vary with significantly different camera positions (e.g., front-on, behind the bowler).
- **Right-Handed Batsman:** The entire logic, including identifying the "front elbow" and "front foot," assumes the subject is a right-handed batsman. The logic would need to be inverted to correctly analyze a left-handed player.
- **Shot Specificity:** All the scoring, feedback, and phase detection heuristics are specifically designed to evaluate a **cover drive**. The tool will not produce a meaningful analysis for other shot types like pull shots, cuts, or defensive strokes.
- **Single Player Focus:** The pose estimation assumes a single, primary subject in the video frame. The presence of multiple people in close proximity might confuse the pose detector.
- **Static Camera:** The analysis assumes the camera is stationary. The logic does not account for panning, zooming, or other camera movements.

## Limitations

- **Heuristic-Based Phase Detection:** The shot is segmented into phases (Stance, Stride, Downswing, etc.) using a heuristic model based on joint velocities. This is not a machine-learning model and may misclassify phases if a player's movements are unconventional or not clearly distinct.
- **Rule-Based Skill Grading:** The final skill grade (Beginner / Intermediate / Advanced) is determined by a simple set of thresholds based on an average of the metric scores. It is not a learned model trained on a large dataset of cricketers.
- **Video Codec Dependency:** The ability to generate a playable annotated video depends on the video codecs (specifically H.264/AVC) being available to the OpenCV library on the system where the code is run.

## Scoring Heuristics

The final 1-10 scores are calculated based on the following rules:

- **Head Position:** The score is based on the average horizontal distance between the player's head and front knee. A smaller distance results in a higher score. The ideal is a distance of 0, and a "poor" distance is defined as 0.15 (15% of the video frame's width).
- **Balance:** This is scored based on the average forward lean of the spine from a perfect vertical line. A smaller angle of lean results in a higher score. The ideal is 0 degrees of lean, and a "poor" lean is defined as 30 degrees.
- **Swing Control:** This score is determined by the maximum angle of the front elbow during the swing. A higher elbow angle is better. The scoring is scaled between a "poor" angle of 90 degrees (score of 1) and an ideal angle of 140 degrees (score of 10).
- **Footwork:** This is scored based on the direction the front foot is pointing. The deviation from an "ideal" angle of 90 degrees (pointing down in the video, towards the off-side) is measured. Zero deviation gets a perfect score, while a deviation of 45 degrees or more is considered "poor."
