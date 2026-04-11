import { useState } from 'react';
import { BarChart3, Zap, TrendingUp } from 'lucide-react';
import { Layout } from './components/Layout';
import { NetworkDashboard } from './pages/NetworkDashboard';
import { PredictionLab } from './pages/PredictionLab';
import { AnalyticsHub } from './pages/AnalyticsHub';
import './index.css';

type PageType = 'dashboard' | 'prediction' | 'analytics';

function App() {
  const [currentPage, setCurrentPage] = useState<PageType>('dashboard');

  const navItems = [
    {
      id: 'dashboard',
      label: 'Network Dashboard',
      icon: '🗺️',
    },
    {
      id: 'prediction',
      label: 'Prediction Lab',
      icon: '⚡',
    },
    {
      id: 'analytics',
      label: 'Analytics Hub',
      icon: '📊',
    },
  ];

  const renderPage = () => {
    switch (currentPage) {
      case 'dashboard':
        return <NetworkDashboard />;
      case 'prediction':
        return <PredictionLab />;
      case 'analytics':
        return <AnalyticsHub />;
      default:
        return <NetworkDashboard />;
    }
  };

  return (
    <Layout
      navItems={navItems}
      currentPage={currentPage}
      onNavigate={(page) => setCurrentPage(page as PageType)}
    >
      {renderPage()}
    </Layout>
  );
}

export default App;
