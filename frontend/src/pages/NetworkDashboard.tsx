import { useEffect, useState } from 'react';
import { motion } from 'framer-motion';
import {
  BarChart,
  Bar,
  PieChart,
  Pie,
  Cell,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from 'recharts';
import { Building2, TrendingUp, Users, Target } from 'lucide-react';
import api from '../api/client';
import {
  StatCard,
  HeroSection,
  Card,
  LoadingSpinner,
  Alert,
} from '../components/UI';
import { StaggerContainer, FadeIn } from '../components/Animations';

interface DashboardData {
  total_hospitals: number;
  total_patients: number;
  global_accuracy: number;
  best_val_accuracy: number;
  avg_f1: number;
  avg_precision: number;
  avg_recall: number;
  num_epochs: number;
  best_epoch: number;
  hospitals: Record<string, any>;
}

const COLORS = ['#2dd4bf', '#38bdf8', '#fbbf24', '#34d399', '#f43f5e'];

export const NetworkDashboard: React.FC = () => {
  const [data, setData] = useState<DashboardData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selectedHospital, setSelectedHospital] = useState<string | null>(null);

  useEffect(() => {
    const loadData = async () => {
      try {
        const response = await api.networkDashboard();
        setData(response.data);
        setError(null);
      } catch (err) {
        setError('Failed to load network dashboard data');
        console.error(err);
      } finally {
        setLoading(false);
      }
    };

    loadData();
  }, []);

  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <LoadingSpinner size="lg" message="Loading network dashboard..." />
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

  const hospitalData = Object.entries(data.hospitals).map(
    ([name, info]: [string, any]) => ({
      name,
      samples: info.total_samples,
    })
  );

  const classDistributionData = Object.entries(
    Object.values(data.hospitals).reduce((acc: any, hospital: any) => {
      Object.entries(hospital.class_distribution || {}).forEach(
        ([cls, count]: [string, any]) => {
          acc[cls] = (acc[cls] || 0) + count;
        }
      );
      return acc;
    }, {})
  ).map(([name, value]) => ({ name, value }));

  return (
    <div className="space-y-8">
      <HeroSection
        title="Network Dashboard"
        description="Live view of the India-wide brain tumor detection federated learning network"
        badges={['Federated Learning', 'Privacy-Preserving', 'Nationwide']}
      />

      {/* Key Metrics */}
      <StaggerContainer>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          <StatCard
            title="Total Hospitals"
            value={data.total_hospitals}
            subtitle="Active contributors"
            icon={<Building2 />}
            index={0}
          />
          <StatCard
            title="Total Patients"
            value={data.total_patients.toLocaleString()}
            subtitle="All cohorts combined"
            icon={<Users />}
            gradient="sky"
            index={1}
          />
          <StatCard
            title="Test Accuracy"
            value={`${data.global_accuracy.toFixed(2)}%`}
            subtitle="Global evaluation"
            icon={<Target />}
            gradient="amber"
            index={2}
          />
          <StatCard
            title="Best Val Accuracy"
            value={`${data.best_val_accuracy.toFixed(2)}%`}
            subtitle={`Epoch ${data.best_epoch}/${data.num_epochs}`}
            icon={<TrendingUp />}
            gradient="cyan"
            index={3}
          />
        </div>
      </StaggerContainer>

      {/* Performance Metrics Row */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <motion.div
          initial={{ opacity: 0, y: 8 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.4 }}
        >
          <Card title="Avg F1-Score">
            <p className="text-3xl font-bold text-teal">
              {data.avg_f1.toFixed(2)}%
            </p>
            <p className="text-xs text-slate mt-2">Macro average</p>
          </Card>
        </motion.div>
        <motion.div
          initial={{ opacity: 0, y: 8 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.5 }}
        >
          <Card title="Avg Precision">
            <p className="text-3xl font-bold text-sky">
              {data.avg_precision.toFixed(2)}%
            </p>
            <p className="text-xs text-slate mt-2">Macro average</p>
          </Card>
        </motion.div>
        <motion.div
          initial={{ opacity: 0, y: 8 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.6 }}
        >
          <Card title="Avg Recall">
            <p className="text-3xl font-bold text-amber">
              {data.avg_recall.toFixed(2)}%
            </p>
            <p className="text-xs text-slate mt-2">Macro average</p>
          </Card>
        </motion.div>
      </div>

      {/* Charts */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <FadeIn delay={0.7}>
          <Card title="Hospital Data Contribution">
            <ResponsiveContainer width="100%" height={300}>
              <PieChart>
                <Pie
                  data={hospitalData}
                  cx="50%"
                  cy="50%"
                  labelLine={false}
                  label={({ name, value }) =>
                    `${name.substring(0, 10)}: ${value}`
                  }
                  outerRadius={80}
                  fill="#8884d8"
                  dataKey="samples"
                >
                  {hospitalData.map((_, index) => (
                    <Cell
                      key={`cell-${index}`}
                      fill={COLORS[index % COLORS.length]}
                    />
                  ))}
                </Pie>
                <Tooltip formatter={(value) => `${value} samples`} />
              </PieChart>
            </ResponsiveContainer>
          </Card>
        </FadeIn>

        <FadeIn delay={0.8}>
          <Card title="Network-Wide Class Distribution">
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={classDistributionData}>
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
                    borderRadius: '8px',
                  }}
                />
                <Bar dataKey="value" fill="#2dd4bf" radius={[8, 8, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </Card>
        </FadeIn>
      </div>

      {/* Hospital Details */}
      <FadeIn delay={0.9}>
        <Card title="Hospital Network Details">
          <div className="space-y-3 max-h-96 overflow-y-auto">
            {Object.entries(data.hospitals).map(
              ([name, hospital]: [string, any]) => (
                <motion.button
                  key={name}
                  whileHover={{ x: 4 }}
                  onClick={() =>
                    setSelectedHospital(selectedHospital === name ? null : name)
                  }
                  className="w-full text-left p-4 rounded-lg border border-slate/20 hover:border-teal/40 hover:bg-slate/10 transition-smooth"
                >
                  <div className="flex items-center justify-between">
                    <div>
                      <p className="font-semibold text-ink">{name}</p>
                      <p className="text-xs text-slate">{hospital.specialty}</p>
                    </div>
                    <p className="text-sm font-bold text-teal">
                      {hospital.total_samples.toLocaleString()}
                    </p>
                  </div>

                  {selectedHospital === name && (
                    <motion.div
                      initial={{ opacity: 0, height: 0 }}
                      animate={{ opacity: 1, height: 'auto' }}
                      exit={{ opacity: 0, height: 0 }}
                      className="mt-3 pt-3 border-t border-slate/20 space-y-2"
                    >
                      <div className="grid grid-cols-2 gap-2 text-xs">
                        {Object.entries(hospital.class_distribution || {}).map(
                          ([cls, count]: [string, any]) => (
                            <div key={cls} className="flex justify-between">
                              <span className="text-slate">{cls}:</span>
                              <span className="font-semibold text-teal">
                                {count}
                              </span>
                            </div>
                          )
                        )}
                      </div>
                    </motion.div>
                  )}
                </motion.button>
              )
            )}
          </div>
        </Card>
      </FadeIn>
    </div>
  );
};
