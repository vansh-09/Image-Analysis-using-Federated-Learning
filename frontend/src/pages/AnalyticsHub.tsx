import { useEffect, useState } from 'react';
import {
  LineChart,
  Line,
  BarChart,
  Bar,
  HeatMap,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from 'recharts';
import { TrendingUp, Award, Zap } from 'lucide-react';
import api from '../api/client';
import { StatCard, HeroSection, Card, LoadingSpinner, Alert } from '../components/UI';
import { StaggerContainer, FadeIn } from '../components/Animations';
import { formatNumber } from '../lib/utils';

interface TrainingEntry {
  epoch: number;
  train_loss: number;
  train_accuracy: number;
  val_loss: number;
  val_accuracy: number;
}

interface AnalyticsData {
  training_history: TrainingEntry[];
  test_metrics: {
    test_accuracy: number;
    avg_precision: number;
    avg_recall: number;
    avg_f1: number;
    per_class: Record<
      string,
      {
        precision: number;
        recall: number;
        f1: number;
        support: number;
      }
    >;
    confusion_matrix?: number[][];
  };
  hospital_contributions: {
    names: string[];
    values: number[];
  };
  class_distribution: Record<string, number>;
}

const COLORS = ['#2dd4bf', '#38bdf8', '#fbbf24', '#34d399', '#f43f5e'];

export const AnalyticsHub: React.FC = () => {
  const [data, setData] = useState<AnalyticsData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const loadAnalytics = async () => {
      try {
        const response = await api.analytics();
        setData(response.data);
        setError(null);
      } catch (err) {
        setError('Failed to load analytics data');
        console.error(err);
      } finally {
        setLoading(false);
      }
    };

    loadAnalytics();
  }, []);

  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <LoadingSpinner size="lg" message="Loading analytics..." />
      </div>
    );
  }

  if (error) {
    return (
      <div className="space-y-4">
        <Alert type="error" message={error} />
      </div>
    );
  }

  if (!data) return null;

  const classMetrics = Object.entries(data.test_metrics.per_class).map(
    ([name, metrics]) => ({
      class: name,
      precision: formatNumber(metrics.precision * 100),
      recall: formatNumber(metrics.recall * 100),
      f1: formatNumber(metrics.f1 * 100),
      support: metrics.support,
    })
  );

  const hospitalContributions = data.hospital_contributions.names.map(
    (name, idx) => ({
      name,
      samples: data.hospital_contributions.values[idx],
    })
  );

  const classDistData = Object.entries(data.class_distribution).map(
    ([name, value]) => ({
      name,
      count: value,
    })
  );

  return (
    <div className="space-y-8">
      <HeroSection
        title="Analytics Hub"
        description="Performance metrics and insights from the federated network"
        badges={['Model Performance', 'Network Insights', 'Real-time Metrics']}
      />

      {/* Key Metrics */}
      <StaggerContainer>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          <StatCard
            title="Test Accuracy"
            value={`${formatNumber(data.test_metrics.test_accuracy * 100)}%`}
            subtitle="Overall performance"
            icon={<Award />}
            index={0}
          />
          <StatCard
            title="Avg Precision"
            value={`${formatNumber(data.test_metrics.avg_precision * 100)}%`}
            subtitle="False positive rate"
            icon={<TrendingUp />}
            gradient="sky"
            index={1}
          />
          <StatCard
            title="Avg Recall"
            value={`${formatNumber(data.test_metrics.avg_recall * 100)}%`}
            subtitle="Sensitivity measure"
            gradient="amber"
            index={2}
          />
          <StatCard
            title="Avg F1-Score"
            value={`${formatNumber(data.test_metrics.avg_f1 * 100)}%`}
            subtitle="Harmonic mean"
            icon={<Zap />}
            gradient="cyan"
            index={3}
          />
        </div>
      </StaggerContainer>

      {/* Training Curves */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <FadeIn delay={0.4}>
          <Card title="Training Loss Over Epochs">
            <ResponsiveContainer width="100%" height={300}>
              <LineChart data={data.training_history}>
                <CartesianGrid
                  strokeDasharray="3 3"
                  stroke="rgba(203, 213, 245, 0.2)"
                />
                <XAxis
                  dataKey="epoch"
                  stroke="rgba(203, 213, 245, 0.5)"
                  label={{
                    value: 'Epoch',
                    position: 'insideBottomRight',
                    offset: -5,
                  }}
                />
                <YAxis
                  stroke="rgba(203, 213, 245, 0.5)"
                  label={{ value: 'Loss', angle: -90, position: 'insideLeft' }}
                />
                <Tooltip
                  contentStyle={{
                    backgroundColor: 'rgba(15, 23, 42, 0.8)',
                    border: '1px solid rgba(203, 213, 245, 0.2)',
                    borderRadius: '8px',
                  }}
                />
                <Legend wrapperStyle={{ paddingTop: '20px' }} />
                <Line
                  type="monotone"
                  dataKey="train_loss"
                  stroke="#2dd4bf"
                  strokeWidth={2}
                  dot={false}
                  name="Training Loss"
                />
                <Line
                  type="monotone"
                  dataKey="val_loss"
                  stroke="#fbbf24"
                  strokeWidth={2}
                  dot={false}
                  name="Validation Loss"
                />
              </LineChart>
            </ResponsiveContainer>
          </Card>
        </FadeIn>

        <FadeIn delay={0.5}>
          <Card title="Training Accuracy Over Epochs">
            <ResponsiveContainer width="100%" height={300}>
              <LineChart data={data.training_history}>
                <CartesianGrid
                  strokeDasharray="3 3"
                  stroke="rgba(203, 213, 245, 0.2)"
                />
                <XAxis
                  dataKey="epoch"
                  stroke="rgba(203, 213, 245, 0.5)"
                  label={{
                    value: 'Epoch',
                    position: 'insideBottomRight',
                    offset: -5,
                  }}
                />
                <YAxis
                  stroke="rgba(203, 213, 245, 0.5)"
                  label={{
                    value: 'Accuracy (%)',
                    angle: -90,
                    position: 'insideLeft',
                  }}
                />
                <Tooltip
                  contentStyle={{
                    backgroundColor: 'rgba(15, 23, 42, 0.8)',
                    border: '1px solid rgba(203, 213, 245, 0.2)',
                    borderRadius: '8px',
                  }}
                  formatter={(value: any) => `${(value * 100).toFixed(2)}%`}
                />
                <Legend wrapperStyle={{ paddingTop: '20px' }} />
                <Line
                  type="monotone"
                  dataKey="train_accuracy"
                  stroke="#38bdf8"
                  strokeWidth={2}
                  dot={false}
                  name="Training Accuracy"
                />
                <Line
                  type="monotone"
                  dataKey="val_accuracy"
                  stroke="#34d399"
                  strokeWidth={2}
                  dot={false}
                  name="Validation Accuracy"
                />
              </LineChart>
            </ResponsiveContainer>
          </Card>
        </FadeIn>
      </div>

      {/* Per-Class Metrics */}
      <FadeIn delay={0.6}>
        <Card title="Performance Metrics by Class">
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-slate/20">
                  <th className="text-left py-3 px-4 text-slate font-semibold">
                    Class
                  </th>
                  <th className="text-right py-3 px-4 text-slate font-semibold">
                    Precision
                  </th>
                  <th className="text-right py-3 px-4 text-slate font-semibold">
                    Recall
                  </th>
                  <th className="text-right py-3 px-4 text-slate font-semibold">
                    F1-Score
                  </th>
                  <th className="text-right py-3 px-4 text-slate font-semibold">
                    Support
                  </th>
                </tr>
              </thead>
              <tbody>
                {classMetrics.map((metric, idx) => (
                  <tr
                    key={metric.class}
                    className="border-b border-slate/10 hover:bg-slate/10 transition-smooth"
                  >
                    <td className="py-3 px-4 font-medium text-ink">
                      {metric.class}
                    </td>
                    <td className="text-right py-3 px-4 text-teal font-semibold">
                      {metric.precision}%
                    </td>
                    <td className="text-right py-3 px-4 text-sky font-semibold">
                      {metric.recall}%
                    </td>
                    <td className="text-right py-3 px-4 text-amber font-semibold">
                      {metric.f1}%
                    </td>
                    <td className="text-right py-3 px-4 text-slate">
                      {metric.support}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>
      </FadeIn>

      {/* Data Contribution & Distribution */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <FadeIn delay={0.7}>
          <Card title="Hospital Data Contribution">
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={hospitalContributions}>
                <CartesianGrid
                  strokeDasharray="3 3"
                  stroke="rgba(203, 213, 245, 0.2)"
                />
                <XAxis
                  dataKey="name"
                  stroke="rgba(203, 213, 245, 0.5)"
                  angle={-45}
                  textAnchor="end"
                  height={80}
                />
                <YAxis stroke="rgba(203, 213, 245, 0.5)" />
                <Tooltip
                  contentStyle={{
                    backgroundColor: 'rgba(15, 23, 42, 0.8)',
                    border: '1px solid rgba(203, 213, 245, 0.2)',
                  }}
                  formatter={(value) => `${value} samples`}
                />
                <Bar dataKey="samples" fill="#2dd4bf" radius={[8, 8, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </Card>
        </FadeIn>

        <FadeIn delay={0.8}>
          <Card title="Class Distribution">
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={classDistData}>
                <CartesianGrid
                  strokeDasharray="3 3"
                  stroke="rgba(203, 213, 245, 0.2)"
                />
                <XAxis dataKey="name" stroke="rgba(203, 213, 245, 0.5)" />
                <YAxis stroke="rgba(203, 213, 245, 0.5)" />
                <Tooltip
                  contentStyle={{
                    backgroundColor: 'rgba(15, 23, 42, 0.8)',
                    border: '1px solid rgba(203, 213, 245, 0.2)',
                  }}
                  formatter={(value) => `${value} samples`}
                />
                <Bar dataKey="count" fill="#38bdf8" radius={[8, 8, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </Card>
        </FadeIn>
      </div>

      {/* Insights Summary */}
      <FadeIn delay={0.9}>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <Card>
            <div className="text-center">
              <p className="text-sm text-slate uppercase tracking-wider mb-2">
                Total Epochs
              </p>
              <p className="text-4xl font-bold text-teal">
                {data.training_history.length}
              </p>
            </div>
          </Card>
          <Card>
            <div className="text-center">
              <p className="text-sm text-slate uppercase tracking-wider mb-2">
                Total Samples
              </p>
              <p className="text-4xl font-bold text-sky">
                {data.hospital_contributions.values
                  .reduce((a, b) => a + b, 0)
                  .toLocaleString()}
              </p>
            </div>
          </Card>
          <Card>
            <div className="text-center">
              <p className="text-sm text-slate uppercase tracking-wider mb-2">
                Classes Detected
              </p>
              <p className="text-4xl font-bold text-amber">
                {Object.keys(data.class_distribution).length}
              </p>
            </div>
          </Card>
        </div>
      </FadeIn>
    </div>
  );
};
