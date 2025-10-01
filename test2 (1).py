"""""
finger_soccer.py
Finger-drawn ball avatar project extended to soccer game:
- Two hands can push balls (index fingertips)
- Left & right screen edges are goals
- Score counting for each side
"""

import cv2
import mediapipe as mp
import numpy as np
import time
import math
import random

# ---------- Config ----------
CAM_INDEX = 0
FRAME_W, FRAME_H = 1280, 720
GRAVITY = 0.9
DT = 1.0
REST = 0.75
FRICTION = 0.995
PINCH_THRESH = 0.04
MAX_RADIUS = 80
MIN_RADIUS = 20

GOAL_HEIGHT = 250
GOAL_WIDTH = 40
# ----------------------------

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

class Ball:
    def __init__(self, x, y, radius, vx=0, vy=0, color=(200,200,255)):
        self.x = x
        self.y = y
        self.r = radius
        self.vx = vx
        self.vy = vy
        self.color = color

    def step(self, w, h):
        self.vy += GRAVITY * DT
        self.x += self.vx * DT
        self.y += self.vy * DT

        # Bounce walls
        if self.x - self.r < 0:
            self.x = self.r
            self.vx = -self.vx * REST
        if self.x + self.r > w:
            self.x = w - self.r
            self.vx = -self.vx * REST
        if self.y + self.r > h:
            self.y = h - self.r
            self.vy = -abs(self.vy) * REST
            self.vx *= FRICTION
            if abs(self.vy) < 1:
                self.vy = 0
        if self.y - self.r < 0:
            self.y = self.r
            self.vy = abs(self.vy) * REST

        self.vx *= 0.999
        self.vy *= 0.999

    def draw(self, frame):
        cv2.circle(frame, (int(self.x), int(self.y)), self.r, self.color, -1, lineType=cv2.LINE_AA)
        # simple eyes
        eye_y = int(self.y - self.r * 0.2)
        eye_x_offset = int(self.r * 0.35)
        eye_r = max(2, int(self.r * 0.12))
        cv2.circle(frame, (int(self.x - eye_x_offset), eye_y), eye_r, (0,0,0), -1)
        cv2.circle(frame, (int(self.x + eye_x_offset), eye_y), eye_r, (0,0,0), -1)
        # smile
        mouth_y = int(self.y + self.r * 0.25)
        cv2.ellipse(frame, (int(self.x), mouth_y), (self.r//2, self.r//4),
                    0, 20, 160, (0,0,0), 2, lineType=cv2.LINE_AA)

def normalized_distance(lm1, lm2):
    return math.hypot(lm1.x - lm2.x, lm1.y - lm2.y)

def landmark_to_pixel(lm, w, h):
    return int(lm.x * w), int(lm.y * h)

def main():
    cap = cv2.VideoCapture(CAM_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)

    hands = mp_hands.Hands(
        model_complexity=0,
        max_num_hands=2,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6
    )

    # One ball to play with
    ball = Ball(FRAME_W//2, FRAME_H//2, 40)

    # Player scores
    score_left = 0
    score_right = 0

    prev_fingers = {}
    prev_times = {}
    finger_vels = {}

    print("Finger Soccer: Use both index fingers to push the ball into goals!")
    print("Left goal = Right player scores | Right goal = Left player scores")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = frame.shape[:2]

        result = hands.process(img_rgb)

        idx_points = []
        if result.multi_hand_landmarks:
            for hi, hand in enumerate(result.multi_hand_landmarks):
                lm_idx = hand.landmark[8]
                idx_pixel = landmark_to_pixel(lm_idx, w, h)
                idx_points.append((hi, idx_pixel))

                mp_drawing.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

                cv2.circle(frame, idx_pixel, 8, (0,255,0), -1)

                # velocity calc per hand
                tnow = time.time()
                if hi in prev_fingers:
                    dt = max(1e-3, tnow - prev_times[hi])
                    vx = (idx_pixel[0] - prev_fingers[hi][0]) / dt
                    vy = (idx_pixel[1] - prev_fingers[hi][1]) / dt
                    finger_vels[hi] = (vx, vy)
                prev_fingers[hi] = idx_pixel
                prev_times[hi] = tnow

        # Push ball with fingers
        for hi, (fx, fy) in idx_points:
            dist = math.hypot(ball.x - fx, ball.y - fy)
            if dist < ball.r:
                vx, vy = finger_vels.get(hi, (0,0))
                ball.vx += vx * 0.05
                ball.vy += vy * 0.05
                overlap = ball.r - dist
                if overlap > 0:
                    ball.x += (ball.x - fx) / (dist+1e-6) * overlap * 0.5
                    ball.y += (ball.y - fy) / (dist+1e-6) * overlap * 0.5
                cv2.circle(frame, (fx, fy), 15, (0,0,255), 2)

        # Step ball physics
        ball.step(w, h)

        # --- Goals ---
        goal_top = h//2 - GOAL_HEIGHT//2
        goal_bot = h//2 + GOAL_HEIGHT//2

        # Left goal
        cv2.rectangle(frame, (0, goal_top), (GOAL_WIDTH, goal_bot), (255,0,0), 2)
        # Right goal
        cv2.rectangle(frame, (w-GOAL_WIDTH, goal_top), (w, goal_bot), (0,0,255), 2)


        # Check scoring
        if ball.x - ball.r <= GOAL_WIDTH and goal_top < ball.y < goal_bot:
            score_right += 1
            ball = Ball(w//2, h//2, 40)
        if ball.x + ball.r >= w - GOAL_WIDTH and goal_top < ball.y < goal_bot:
            score_left += 1
            ball = Ball(w//2, h//2, 40)

        # Draw ball
        ball.draw(frame)

        # Display score
        cv2.putText(frame, f"Left: {score_left}", (50,50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255,255,255), 3)
        cv2.putText(frame, f"Right: {score_right}", (w-250,50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255,255,255), 3)

        cv2.imshow("Finger Soccer", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key == ord('r'):
            score_left, score_right = 0, 0
            ball = Ball(w//2, h//2, 40)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
    