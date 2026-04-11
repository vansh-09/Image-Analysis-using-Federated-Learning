export const cn = (...classes: (string | undefined | null | false)[]) => {
  return classes.filter(Boolean).join(' ');
};

export const formatNumber = (num: number, decimals = 2): string => {
  return num.toFixed(decimals);
};

export const formatPercent = (num: number): string => {
  return `${(num * 100).toFixed(2)}%`;
};

export const formatDate = (date: string | Date): string => {
  const d = typeof date === 'string' ? new Date(date) : date;
  return d.toLocaleDateString('en-US', {
    year: 'numeric',
    month: 'short',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  });
};

export const calculateImageStats = async (
  file: File
): Promise<{
  type: string;
  size: string;
  dimensions?: string;
}> => {
  return new Promise((resolve) => {
    const reader = new FileReader();
    reader.onload = (e) => {
      const img = new Image();
      img.onload = () => {
        resolve({
          type: file.type,
          size: `${(file.size / 1024).toFixed(2)} KB`,
          dimensions: `${img.width} x ${img.height}`,
        });
      };
      img.src = e.target?.result as string;
    };
    reader.readAsDataURL(file);
  });
};

export const validateImage = (file: File): string | null => {
  const validTypes = ['image/png', 'image/jpeg', 'image/jpg'];
  const maxSize = 16 * 1024 * 1024; // 16MB

  if (!validTypes.includes(file.type)) {
    return 'Please upload a PNG or JPEG image';
  }

  if (file.size > maxSize) {
    return 'File size exceeds 16MB limit';
  }

  return null;
};

export const downloadJSON = (data: any, filename: string) => {
  const json = JSON.stringify(data, null, 2);
  const blob = new Blob([json], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = filename;
  link.click();
  URL.revokeObjectURL(url);
};
