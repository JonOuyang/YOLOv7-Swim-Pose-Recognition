# YOLOv7 Keypoints

YOLOv7 Pose Estimation will return an array of 56 coordinates. The first 5 coordinates are information relating to the bounding box and its confidence. That information is irrelevant to this project. The next 51 arrays are information about the actual joints.

The information is split into groups of 3. The first number represents x coordinate, the second number represents y coordinate, the third number representes confidence level in position of that joint. (x, y, conf). For a visual digram of joint keypoints look at the [Visual.png file](Visual.png)

### Original order of the joints (by index)
0. nose
1. left eye (right for viewer)
2. right eye (left for viewer)
3. left ear (right for viewer)
4. right ear (left for viewer)
5. left shoulder (right for viewer)
6. right shoulder (left for viewer)
7. left elbow (right for viewer)
8. right elbow (left for viewer)
9. left wrist (right for viewer)
10. right wrist (left for viewer)
11. left hip (right for viewer)
12. right hip (left for viewer)
13. left knee (right for viewer)
14. right knee (left for viewer)
15. left ankle (right for viewer)
16. right ankle (left for viewer)

(Models displayed significant improvements with feature engineering; when the nose, eye, and ear coordinates are removed from consideration)
First 5 indices were removes (indices 0-4)

### Modified order after removal of face keypoints:
0. left shoulder (right for viewer)
1. right shoulder (left for viewer)
2. left elbow (right for viewer)
3. right elbow (left for viewer)
4. left wrist (right for viewer)
5. right wrist (left for viewer)
6. left hip (right for viewer)
7. right hip (left for viewer)
8. left knee (right for viewer)
9. right knee (left for viewer)
10. left ankle (right for viewer)
11. right ankle (left for viewer)
