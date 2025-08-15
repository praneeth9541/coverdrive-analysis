import cv2
import mediapipe as mp
import numpy as np

def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle


def generate_evaluation(metrics_history):
    if not metrics_history:
        return {}

    
    head_dists = [m['head_knee_dist'] for m in metrics_history]
    avg_head_dist = np.mean(head_dists) if head_dists else 0.15
    head_perf_ratio = max(0, 1 - (avg_head_dist / 0.15))
    head_score = 1 + 9 * head_perf_ratio 
    head_feedback = "Excellent head stability." if head_score > 8.5 else "Keep your head over your front knee."

    spine_angles = [m['spine_angle'] for m in metrics_history]
    avg_spine_angle = np.mean(spine_angles) if spine_angles else 30
    balance_perf_ratio = max(0, 1 - (avg_spine_angle / 30))
    balance_score = 1 + 9 * balance_perf_ratio 
    balance_feedback = "Good, stable core." if balance_score > 8.5 else "Maintain a more stable posture."

    elbow_angles = [m['elbow_angle'] for m in metrics_history]
    max_elbow_angle = np.max(elbow_angles) if elbow_angles else 90
    swing_perf_ratio = (max_elbow_angle - 90) / (140 - 90)
    swing_perf_ratio = max(0, min(1, swing_perf_ratio))
    swing_score = 1 + 9 * swing_perf_ratio 
    swing_feedback = "Great elbow elevation." if swing_score > 8.5 else "Raise your front elbow higher."
    
    foot_angles = [m['foot_angle'] for m in metrics_history if 'foot_angle' in m and m['foot_angle'] is not None]
    if foot_angles:
        avg_foot_angle = np.mean(foot_angles)
        deviation = abs(avg_foot_angle - 90)
        footwork_perf_ratio = max(0, 1 - (deviation / 45))
        footwork_score = 1 + 9 * footwork_perf_ratio
        footwork_feedback = "Good front foot placement." if footwork_score > 8.5 else "Point your front foot more towards the off-side."
    else:
        footwork_score = 1 
        footwork_feedback = "Foot angle could not be determined."

    average_score = np.mean([head_score, balance_score, swing_score, footwork_score])
    
    if average_score > 8.5:
        skill_grade = "Advanced"
    elif average_score > 6.0:
        skill_grade = "Intermediate"
    else:
        skill_grade = "Beginner"

    evaluation = {
        "Overall Skill Grade": skill_grade, 
        "Average Score": round(average_score, 2),
        "breakdown": {
            "Head Position": {"score": round(head_score, 1), "feedback": head_feedback},
            "Balance": {"score": round(balance_score, 1), "feedback": balance_feedback},
            "Swing Control": {"score": round(swing_score, 1), "feedback": swing_feedback},
            "Footwork": {"score": round(footwork_score, 1), "feedback": footwork_feedback}
        }
    }
    return evaluation


def analyze_video(video_path, output_path):
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose()
    mp_drawing = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    if not out.isOpened():
        print("Error: VideoWriter failed to open. Check that the path is correct and codecs are available.")
        return None, {"error": "Failed to create output video."}
    TEXT_COLOR = (0, 255, 255) 

    phase = "Stance"
    prev_landmarks = None
    frame_counter = 0

    metrics_history = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_counter += 1
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)

        try:
            landmarks = results.pose_landmarks.landmark
            left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
            elbow_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
            right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
            left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
            right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
            shoulder_mid = np.mean([left_shoulder, right_shoulder], axis=0)
            hip_mid = np.mean([left_hip, right_hip], axis=0)
            vertical_point = [hip_mid[0], hip_mid[1] - 1]
            spine_angle = calculate_angle(shoulder_mid, hip_mid, vertical_point)
            nose = [landmarks[mp_pose.PoseLandmark.NOSE.value].x, landmarks[mp_pose.PoseLandmark.NOSE.value].y]
            left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
            head_knee_dist = abs(nose[0] - left_knee[0])
            left_heel = [landmarks[mp_pose.PoseLandmark.LEFT_HEEL.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HEEL.value].y]
            left_foot_index = [landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].x, landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].y]
            foot_angle_rad = np.arctan2(left_foot_index[1] - left_heel[1], left_foot_index[0] - left_heel[0])
            foot_angle_deg = 180 - np.degrees(foot_angle_rad)

            if prev_landmarks:
                left_ankle_pos = np.array([landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y])
                prev_ankle_pos = np.array([prev_landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, prev_landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y])
                ankle_velocity = np.linalg.norm(left_ankle_pos - prev_ankle_pos)
                
                left_wrist_pos = np.array([landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y])
                prev_wrist_pos = np.array([prev_landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, prev_landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y])
                wrist_velocity = np.linalg.norm(left_wrist_pos - prev_wrist_pos)

                if phase == "Stance" and ankle_velocity > 0.01:
                    phase = "Stride"
                elif phase == "Stride" and ankle_velocity < 0.005 and wrist_velocity > 0.02:
                    phase = "Downswing"
                elif phase == "Downswing" and wrist_velocity < 0.03:
                    phase = "Follow-through"

            prev_landmarks = landmarks

            metrics_history.append({
                'elbow_angle': elbow_angle, 
                'spine_angle': spine_angle, 
                'head_knee_dist': head_knee_dist,
                'foot_angle': foot_angle_deg,
                'phase': phase 
            })
            
            head_feedback = "HEAD OK" if head_knee_dist <= 0.05 else "HEAD NOT OVER KNEE"
            cv2.putText(frame, head_feedback, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0) if head_feedback == "HEAD OK" else (0, 0, 255), 2, cv2.LINE_AA)

            elbow_pos = tuple(np.multiply(left_elbow, [frame_width, frame_height]).astype(int))
            cv2.putText(frame, f"Elbow: {int(elbow_angle)}", elbow_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.7, TEXT_COLOR, 2, cv2.LINE_AA)
            
            foot_pos = tuple(np.multiply(left_foot_index, [frame_width, frame_height]).astype(int))
            cv2.putText(frame, f"Foot: {int(foot_angle_deg)} deg", foot_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.7, TEXT_COLOR, 2, cv2.LINE_AA)
            
            cv2.putText(frame, f"Phase: {phase}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, TEXT_COLOR, 2, cv2.LINE_AA)


        except Exception as e:
            pass

        if results.pose_landmarks:
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        out.write(frame)

    cap.release()
    out.release()
    
    final_evaluation = generate_evaluation(metrics_history)
    return output_path, final_evaluation