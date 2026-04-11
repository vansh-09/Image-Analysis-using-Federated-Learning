import { useState } from 'react';
import { motion } from 'framer-motion';
import { Menu, X } from 'lucide-react';
import { cn } from '../lib/utils';

interface NavItem {
  id: string;
  label: string;
  icon: React.ReactNode;
}

interface LayoutProps {
  navItems: NavItem[];
  currentPage: string;
  onNavigate: (page: string) => void;
  children: React.ReactNode;
}

export const Layout: React.FC<LayoutProps> = ({
  navItems,
  currentPage,
  onNavigate,
  children,
}) => {
  const [sidebarOpen, setSidebarOpen] = useState(true);

  return (
    <div className="flex h-screen bg-paper overflow-hidden">
      {/* Sidebar */}
      <motion.aside
        animate={{ width: sidebarOpen ? 256 : 80 }}
        transition={{ duration: 0.3 }}
        className="glass border-r border-slate/20 flex flex-col"
      >
        <div className="p-6 border-b border-slate/20 flex items-center justify-between">
          {sidebarOpen && (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="font-fraunces text-xl font-bold gradient-text"
            >
              MediSync
            </motion.div>
          )}
          <motion.button
            whileHover={{ scale: 1.1 }}
            whileTap={{ scale: 0.95 }}
            onClick={() => setSidebarOpen(!sidebarOpen)}
            className="p-2 hover:bg-slate/20 rounded-lg transition-smooth"
          >
            {sidebarOpen ? <X size={20} /> : <Menu size={20} />}
          </motion.button>
        </div>

        <nav className="flex-1 overflow-y-auto p-4 space-y-2">
          {navItems.map((item) => (
            <motion.button
              key={item.id}
              whileHover={{ x: 4 }}
              onClick={() => onNavigate(item.id)}
              className={cn(
                'w-full flex items-center gap-3 px-4 py-3 rounded-lg transition-smooth',
                currentPage === item.id
                  ? 'bg-teal/20 border border-teal/40 text-teal'
                  : 'text-slate hover:bg-slate/10 border border-transparent'
              )}
            >
              <span className="text-xl flex-shrink-0">{item.icon}</span>
              {sidebarOpen && (
                <motion.span
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  exit={{ opacity: 0 }}
                  className="text-sm font-medium"
                >
                  {item.label}
                </motion.span>
              )}
            </motion.button>
          ))}
        </nav>

        <div className="p-4 border-t border-slate/20 text-xs text-slate/60 text-center">
          {sidebarOpen && <p>v1.0.0</p>}
        </div>
      </motion.aside>

      {/* Main Content */}
      <div className="flex-1 flex flex-col overflow-hidden">
        {/* Header */}
        <header className="glass border-b border-slate/20 px-8 py-4 flex items-center justify-between">
          <h1 className="text-2xl font-bold text-ink">MediSync FL India</h1>
          <div className="flex items-center gap-4">
            <div className="w-2 h-2 bg-emerald-500 rounded-full animate-pulse" />
            <span className="text-xs text-slate">Live Network Status</span>
          </div>
        </header>

        {/* Content */}
        <main className="flex-1 overflow-y-auto">
          <div className="p-8">{children}</div>
        </main>
      </div>
    </div>
  );
};
