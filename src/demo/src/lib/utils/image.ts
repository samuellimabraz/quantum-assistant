const MAX_DIMENSION = 640;

export async function resizeImageForInference(
    imageSource: string
): Promise<string> {
    return new Promise((resolve, reject) => {
        const img = new Image();
        img.crossOrigin = 'anonymous';

        img.onload = () => {
            let { width, height } = img;

            if (width <= MAX_DIMENSION && height <= MAX_DIMENSION) {
                if (imageSource.startsWith('data:')) {
                    resolve(imageSource.split(',')[1]);
                } else {
                    const canvas = document.createElement('canvas');
                    canvas.width = width;
                    canvas.height = height;
                    const ctx = canvas.getContext('2d');
                    if (!ctx) {
                        reject(new Error('Failed to get canvas context'));
                        return;
                    }
                    ctx.drawImage(img, 0, 0);
                    const dataUrl = canvas.toDataURL('image/jpeg', 0.95);
                    resolve(dataUrl.split(',')[1]);
                }
                return;
            }

            const scale = Math.min(MAX_DIMENSION / width, MAX_DIMENSION / height);
            const newWidth = Math.round(width * scale);
            const newHeight = Math.round(height * scale);

            const canvas = document.createElement('canvas');
            canvas.width = newWidth;
            canvas.height = newHeight;

            const ctx = canvas.getContext('2d');
            if (!ctx) {
                reject(new Error('Failed to get canvas context'));
                return;
            }

            ctx.imageSmoothingEnabled = true;
            ctx.imageSmoothingQuality = 'high';
            ctx.drawImage(img, 0, 0, newWidth, newHeight);

            const dataUrl = canvas.toDataURL('image/jpeg', 0.95);
            resolve(dataUrl.split(',')[1]);
        };

        img.onerror = () => {
            reject(new Error('Failed to load image'));
        };

        img.src = imageSource;
    });
}

export async function fetchAndResizeImage(url: string): Promise<string> {
    try {
        const response = await fetch(url);
        const blob = await response.blob();
        const dataUrl = await new Promise<string>((resolve) => {
            const reader = new FileReader();
            reader.onloadend = () => resolve(reader.result as string);
            reader.readAsDataURL(blob);
        });
        return resizeImageForInference(dataUrl);
    } catch (error) {
        console.error('Failed to fetch and resize image:', error);
        throw error;
    }
}

