// Format seconds into MM:SS
export function formatTime(seconds) {
    const minutes = Math.floor(seconds / 60);
    const remaining = Math.floor(seconds % 60);
    return `${String(minutes).padStart(2, "0")}:${String(remaining).padStart(2, "0")}`;
  }
  
  // Format file information (takes name and size)
  export function getFileInfo(filename, fileSize) {
    const extension = filename.split('.').pop().toLowerCase();
  
    let sizeStr;
    if (fileSize < 1024) {
      sizeStr = `${fileSize} bytes`;
    } else if (fileSize < 1024 * 1024) {
      sizeStr = `${(fileSize / 1024).toFixed(1)} KB`;
    } else {
      sizeStr = `${(fileSize / (1024 * 1024)).toFixed(1)} MB`;
    }
  
    const currentTime = new Date().toLocaleString();
  
    return `File: ${filename}\nSize: ${sizeStr}\nType: ${extension.toUpperCase()}\nUploaded: ${currentTime}`;
  }
  
  // Create an emotion description from emotion name and probability
  export function createEmotionDescription(emotion, probability) {
    const capitalized = emotion.charAt(0).toUpperCase() + emotion.slice(1);
  
    if (probability > 0.8) {
      return `Strong indication of ${capitalized}`;
    } else if (probability > 0.6) {
      return `Clear presence of ${capitalized}`;
    } else if (probability > 0.4) {
      return `Moderate signs of ${capitalized}`;
    } else if (probability > 0.2) {
      return `Slight hints of ${capitalized}`;
    } else {
      return `Very little ${capitalized} detected`;
    }
  }
  
  // Validate an uploaded audio file (checks extension and size)
  export function validateAudioFile(file) {
    const supportedFormats = ['wav', 'mp3', 'ogg', 'flac', 'm4a'];
  
    if (!file) return false;
  
    const extension = file.name.split('.').pop().toLowerCase();
    const maxSize = 50 * 1024 * 1024; // 50 MB
  
    return supportedFormats.includes(extension) && file.size <= maxSize;
  }
  