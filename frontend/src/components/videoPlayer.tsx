import React from "react";

type VideoPlayerProps = {
    videoId: string;  // the ID from GridFS
  };
  
  const VideoPlayer: React.FC<VideoPlayerProps> = ({ videoId }: VideoPlayerProps) => {
    return (
      <video controls width="100%">
        <source src={`http://127.0.0.1:8000/download/video/byid/${videoId}`} type="video/mp4" />
      </video>
    );
  };
  
  export default VideoPlayer;
  