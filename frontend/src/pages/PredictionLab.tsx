import { useEffect, useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  LineChart,
  Line,
} from 'recharts';
import { Upload, Download, Zap, AlertCircle } from 'lucide-react';
import api from '../api/client';
import {
  StatCard,
  HeroSection,
  Card,
  Button,
  LoadingSpinner,
  Alert,
} from '../components/UI';
import { FadeIn, StaggerContainer } from '../components/Animations';
import { validateImage, calculateImageStats, formatNumber } from '../lib/utils';

interface PredictionResult {
  predicted_class: string;
  confidence: number;
  timestamp: string;
  image_properties: {
    resolution: string;
    brightness: number;
    contrast: number;
  };
  all_predictions: Record<string, number>;
  device: string;
  model_info: {
    trained_at: string;
    test_accuracy: number;
  };
}

export const PredictionLab: React.FC = () => {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [result, setResult] = useState<PredictionResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [imageStats, setImageStats] = useState<any>(null);
  const [modelInfo, setModelInfo] = useState<any>(null);
  const [confidenceThreshold] = useState(70);
  const [predictionsHistory, setPredictionsHistory] = useState<
    PredictionResult[]
  >([]);

  useEffect(() => {
    const loadModelInfo = async () => {
      try {
        const response = await api.modelInfo();
        setModelInfo(response.data);
      } catch (err) {
        console.error('Failed to load model info', err);
      }
    };

    loadModelInfo();
  }, []);

  const handleFileSelect = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    const validationError = validateImage(file);
    if (validationError) {
      setError(validationError);
      return;
    }

    setSelectedFile(file);
    setError(null);

    // Generate preview
    const reader = new FileReader();
    reader.onload = (e) => setPreview(e.target?.result as string);
    reader.readAsDataURL(file);

    // Calculate stats
    const stats = await calculateImageStats(file);
    setImageStats(stats);
  };

  const handlePredict = async () => {
    if (!selectedFile) return;

    setLoading(true);
    setError(null);

    try {
      const response = await api.predict(selectedFile);
      const prediction = response.data;
      setResult(prediction);
      setPredictionsHistory([prediction, ...predictionsHistory].slice(0, 5));
    } catch (err: any) {
      setError(
        err.response?.data?.error || 'Prediction failed. Please try again.'
      );
    } finally {
      setLoading(false);
    }
  };

  const downloadReport = async () => {
    if (!selectedFile) return;

    setLoading(true);
    try {
      const response = await api.predictPDF(selectedFile);
      const blob = response.data;
      const url = URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.download = `prediction_report_${Date.now()}.pdf`;
      link.click();
      URL.revokeObjectURL(url);
    } catch (err) {
      setError('Failed to download PDF report');
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  const sortedPredictions = result
    ? Object.entries(result.all_predictions)
        .sort(([, a], [, b]) => b - a)
        .slice(0, 4)
        .map(([label, prob]) => ({
          class: label,
          probability: (prob * 100).toFixed(2),
        }))
    : [];

  return (
    <div className="space-y-8">
      <HeroSection
        title="Prediction Lab"
        description="Upload an MRI scan to get instant predictions from the global federated model"
        badges={['Real-time Analysis', 'Federated ResNet18', 'GPU Accelerated']}
      />

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Upload Section */}
        <div className="lg:col-span-2 space-y-4">
          <FadeIn>
            <Card title="Upload MRI Scan">
              {!selectedFile ? (
                <motion.div
                  whileHover={{ scale: 1.02 }}
                  className="border-2 border-dashed border-teal/40 rounded-2xl p-12 text-center cursor-pointer hover:border-teal/60 transition-colors"
                  onClick={() => document.getElementById('file-input')?.click()}
                >
                  <motion.div
                    animate={{ y: [-4, 4, -4] }}
                    transition={{ duration: 2, repeat: Infinity }}
                    className="mb-4"
                  >
                    <Upload className="w-12 h-12 mx-auto text-teal/60" />
                  </motion.div>
                  <p className="text-ink font-semibold mb-1">
                    Click to upload or drag and drop
                  </p>
                  <p className="text-sm text-slate">PNG or JPEG • Up to 16MB</p>
                  <input
                    id="file-input"
                    type="file"
                    accept="image/png,image/jpeg"
                    onChange={handleFileSelect}
                    className="hidden"
                  />
                </motion.div>
              ) : (
                <motion.div
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  className="space-y-4"
                >
                  {preview && (
                    <div className="relative rounded-xl overflow-hidden">
                      <img
                        src={preview}
                        alt="Preview"
                        className="w-full h-auto rounded-xl border border-slate/20"
                      />
                      <motion.div
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        className="absolute inset-0 bg-gradient-to-b from-transparent to-paper/40"
                      />
                    </div>
                  )}

                  {imageStats && (
                    <div className="grid grid-cols-3 gap-2 text-xs">
                      <div className="glass rounded-lg p-2 text-center">
                        <p className="text-slate">Size</p>
                        <p className="font-semibold text-ink">
                          {imageStats.size}
                        </p>
                      </div>
                      <div className="glass rounded-lg p-2 text-center">
                        <p className="text-slate">Dimensions</p>
                        <p className="font-semibold text-ink">
                          {imageStats.dimensions}
                        </p>
                      </div>
                      <div className="glass rounded-lg p-2 text-center">
                        <p className="text-slate">Format</p>
                        <p className="font-semibold text-ink">
                          {imageStats.type.split('/')[1].toUpperCase()}
                        </p>
                      </div>
                    </div>
                  )}

                  <Button
                    onClick={handlePredict}
                    loading={loading}
                    variant="primary"
                    className="w-full"
                  >
                    <Zap size={18} />
                    {loading ? 'Analyzing...' : 'Run Prediction'}
                  </Button>

                  <Button
                    onClick={() => {
                      setSelectedFile(null);
                      setPreview(null);
                      setResult(null);
                      setImageStats(null);
                    }}
                    variant="outline"
                    className="w-full"
                  >
                    Choose Different Image
                  </Button>
                </motion.div>
              )}
            </Card>
          </FadeIn>

          {/* Tips Card */}
          {!selectedFile && (
            <FadeIn delay={0.2}>
              <Card title="Tips for Best Results">
                <ul className="space-y-2 text-sm text-slate">
                  <li>✓ Use axial MRI slices with clear tumor boundaries</li>
                  <li>✓ Ensure the scan is well-lit and not cropped</li>
                  <li>✓ PNG or high-quality JPEG works best</li>
                  <li>✓ Image resolution should be at least 224x224 pixels</li>
                </ul>
              </Card>
            </FadeIn>
          )}
        </div>

        {/* Model Info Sidebar */}
        <FadeIn delay={0.3}>
          <Card title="Model Snapshot">
            {modelInfo ? (
              <div className="space-y-4 text-sm">
                <div>
                  <p className="text-slate">Status</p>
                  <p className="font-semibold text-teal">Ready</p>
                </div>
                <div>
                  <p className="text-slate">Trained</p>
                  <p className="font-semibold text-ink text-xs">
                    {modelInfo.trained_at?.split('T')[0] || 'N/A'}
                  </p>
                </div>
                <div>
                  <p className="text-slate">Epochs</p>
                  <p className="font-semibold text-ink">
                    {modelInfo.best_epoch}/{modelInfo.num_epochs}
                  </p>
                </div>
                <div className="border-t border-slate/20 pt-4">
                  <p className="text-slate">Test Accuracy</p>
                  <p className="text-2xl font-bold gradient-text">
                    {modelInfo.metrics?.test_accuracy
                      ? `${(modelInfo.metrics.test_accuracy * 100).toFixed(2)}%`
                      : 'N/A'}
                  </p>
                </div>
                <div>
                  <p className="text-slate">Avg F1</p>
                  <p className="font-semibold text-ink">
                    {modelInfo.metrics?.avg_f1
                      ? `${(modelInfo.metrics.avg_f1 * 100).toFixed(2)}%`
                      : 'N/A'}
                  </p>
                </div>
              </div>
            ) : (
              <LoadingSpinner size="sm" />
            )}
          </Card>
        </FadeIn>
      </div>

      {/* Predictions Results */}
      <AnimatePresence>
        {error && (
          <motion.div
            initial={{ opacity: 0, y: -8 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -8 }}
          >
            <Alert
              type="error"
              message={error}
              onClose={() => setError(null)}
            />
          </motion.div>
        )}
      </AnimatePresence>

      {result && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          className="space-y-6"
        >
          {/* Main Result */}
          <FadeIn>
            <div className="glass rounded-2xl p-8 border border-teal/40">
              <div className="flex items-center justify-between mb-6">
                <div>
                  <p className="text-sm text-slate uppercase tracking-wider">
                    Prediction Result
                  </p>
                  <p className="text-4xl font-bold text-teal mt-2">
                    {result.predicted_class}
                  </p>
                </div>
                <motion.div
                  animate={{ scale: [1, 1.1, 1] }}
                  transition={{ duration: 2, repeat: Infinity }}
                  className="text-5xl"
                >
                  {result.confidence > 0.8 ? '✓' : '!'}
                </motion.div>
              </div>

              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                <div>
                  <p className="text-xs text-slate uppercase">Confidence</p>
                  <p className="text-2xl font-bold text-sky mt-1">
                    {formatNumber(result.confidence * 100)}%
                  </p>
                </div>
                <div>
                  <p className="text-xs text-slate uppercase">Resolution</p>
                  <p className="text-lg font-semibold text-amber mt-1">
                    {result.image_properties.resolution}
                  </p>
                </div>
                <div>
                  <p className="text-xs text-slate uppercase">Brightness</p>
                  <p className="text-lg font-semibold text-emerald-400 mt-1">
                    {formatNumber(result.image_properties.brightness)}
                  </p>
                </div>
                <div>
                  <p className="text-xs text-slate uppercase">Device</p>
                  <p className="text-lg font-semibold text-pink-400 mt-1">
                    {result.device}
                  </p>
                </div>
              </div>

              {result.confidence < confidenceThreshold / 100 && (
                <motion.div
                  initial={{ opacity: 0, y: 8 }}
                  animate={{ opacity: 1, y: 0 }}
                  className="mt-6 p-4 bg-amber/20 border border-amber/40 rounded-lg flex gap-3"
                >
                  <AlertCircle className="w-5 h-5 text-amber flex-shrink-0 mt-0.5" />
                  <p className="text-sm text-amber-100">
                    Confidence is below threshold ({confidenceThreshold}%).
                    Consider manual review.
                  </p>
                </motion.div>
              )}
            </div>
          </FadeIn>

          {/* Charts */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <FadeIn delay={0.2}>
              <Card title="Prediction Confidence">
                <ResponsiveContainer width="100%" height={250}>
                  <BarChart data={sortedPredictions}>
                    <CartesianGrid
                      strokeDasharray="3 3"
                      stroke="rgba(203, 213, 245, 0.2)"
                    />
                    <XAxis dataKey="class" stroke="rgba(203, 213, 245, 0.5)" />
                    <YAxis stroke="rgba(203, 213, 245, 0.5)" />
                    <Tooltip
                      contentStyle={{
                        backgroundColor: 'rgba(15, 23, 42, 0.8)',
                        border: '1px solid rgba(203, 213, 245, 0.2)',
                      }}
                      formatter={(value) => `${value}%`}
                    />
                    <Bar
                      dataKey="probability"
                      fill="#2dd4bf"
                      radius={[8, 8, 0, 0]}
                    />
                  </BarChart>
                </ResponsiveContainer>
              </Card>
            </FadeIn>

            <FadeIn delay={0.3}>
              <Card title="All Predictions">
                <div className="space-y-3">
                  {Object.entries(result.all_predictions)
                    .sort(([, a], [, b]) => b - a)
                    .map(([label, prob], idx) => (
                      <motion.div
                        key={label}
                        initial={{ opacity: 0, x: -8 }}
                        animate={{ opacity: 1, x: 0 }}
                        transition={{ delay: idx * 0.1 }}
                        className="flex items-center gap-3"
                      >
                        <div className="flex-1">
                          <div className="flex justify-between mb-1">
                            <span className="text-sm font-medium text-ink">
                              {label}
                            </span>
                            <span className="text-xs font-semibold text-teal">
                              {formatNumber(prob * 100)}%
                            </span>
                          </div>
                          <motion.div
                            initial={{ width: 0 }}
                            animate={{ width: `${prob * 100}%` }}
                            transition={{ duration: 0.6, delay: idx * 0.05 }}
                            className="h-2 bg-gradient-to-r from-teal to-sky rounded-full"
                          />
                        </div>
                      </motion.div>
                    ))}
                </div>
              </Card>
            </FadeIn>
          </div>

          {/* Action Buttons */}
          <div className="flex gap-4 justify-end">
            <Button variant="outline" onClick={downloadReport}>
              <Download size={18} />
              Download Report
            </Button>
          </div>
        </motion.div>
      )}

      {/* Recent Predictions */}
      {predictionsHistory.length > 0 && (
        <FadeIn delay={0.5}>
          <Card title="Recent Predictions">
            <div className="space-y-2 max-h-48 overflow-y-auto">
              {predictionsHistory.map((pred, idx) => (
                <motion.div
                  key={idx}
                  initial={{ opacity: 0, x: -8 }}
                  animate={{ opacity: 1, x: 0 }}
                  className="flex items-center justify-between p-3 rounded-lg bg-slate/10 border border-slate/20"
                >
                  <div>
                    <p className="font-semibold text-ink">
                      {pred.predicted_class}
                    </p>
                    <p className="text-xs text-slate">
                      {new Date(pred.timestamp).toLocaleTimeString()}
                    </p>
                  </div>
                  <p className="text-lg font-bold text-teal">
                    {formatNumber(pred.confidence * 100)}%
                  </p>
                </motion.div>
              ))}
            </div>
          </Card>
        </FadeIn>
      )}
    </div>
  );
};
