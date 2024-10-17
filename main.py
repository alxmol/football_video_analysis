from utils import read_video, save_video
from team_assigner import TeamAssigner
from trackers import Tracker
from player_ball_assigner import PlayerBallAssigner
from camera_move import CameraMovement
import cv2
from transformer import Transformer
import numpy as np
from speed_distance_est import SpeedDistanceEstimator


def main():
    video_frames = read_video('input_videos/08fd33_4.mp4')
    
    tracker = Tracker('training/models/best.pt')
    
    tracks = tracker.get_object_tracks(video_frames, read_from_stub=True, stub_path='stubs/track_stubs.pkl')
    
    tracker.add_position_to_tracks(tracks)
    
    camera_movement = CameraMovement(video_frames[0])
    camera_movement_per_frame = camera_movement.get_camera_movement(video_frames, read_from_stub=True, stub_path='stubs/camera_movement_stubs.pkl')
    
    camera_movement.add_adjust_positions_to_tracks(tracks, camera_movement_per_frame)
    
    transformer = Transformer()
    transformer.add_transformed_position_to_tracks(tracks)
    
    tracks["ball"] = tracker.interpolate_ball_coords(tracks["ball"])
    
    speed_distance_estimator = SpeedDistanceEstimator()
    speed_distance_estimator.add_speed_distance_to_tracks(tracks)
    
    team_assigner = TeamAssigner()
    team_assigner.assign_team_color(video_frames[0], tracks['players'][0])
    
    for frame_num, player_track in enumerate(tracks['players']):
        for player_id, track in player_track.items():
            team = team_assigner.get_player_team(video_frames[frame_num], track['bbox'], player_id)
            
            tracks['players'][frame_num][player_id]['team'] = team
            tracks['players'][frame_num][player_id]['team_color'] = team_assigner.team_colors[team]
            
    player_assigner = PlayerBallAssigner()
    team_posession = []
    for frame_num, player_track in enumerate(tracks['players']):
        ball_bbox = tracks['ball'][frame_num][1]['bbox']
        assigned_player = player_assigner.assign_player_to_ball(player_track, ball_bbox)
        
        if assigned_player != -1:
            tracks['players'][frame_num][assigned_player]['has_ball'] = True
            team_posession.append(tracks['players'][frame_num][assigned_player]['team'])
        else:
            team_posession.append(team_posession[-1])
    team_posession = np.array(team_posession)
    
    output_video_frames = tracker.draw_annotations(video_frames, tracks, team_posession)
    
    output_video_frames = camera_movement.draw_camera_movement(output_video_frames, camera_movement_per_frame)
    
    speed_distance_estimator.draw_speed_distance(output_video_frames, tracks)
    
    save_video(output_video_frames, 'output_videos/output_video.avi')
    
if __name__ == '__main__':
    main()
    
    #commit changes