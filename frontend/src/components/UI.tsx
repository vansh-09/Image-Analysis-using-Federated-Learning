import { ReactNode } from 'react';
import { motion } from 'framer-motion';
import { cn } from '../lib/utils';
import { FadeIn } from './Animations';

interface StatCardProps {
  title: string;
  value: string | number;
  subtitle?: string;
  icon?: ReactNode;
  gradient?: 'teal' | 'sky' | 'amber' | 'cyan';
  index?: number;
}

const gradientMap = {
  teal: 'from-teal/20 to-cyan-500/20',
  sky: 'from-sky/20 to-blue-500/20',
  amber: 'from-amber/20 to-orange-500/20',
  cyan: 'from-cyan-500/20 to-teal/20',
};

export const StatCard: React.FC<StatCardProps> = ({
  title,
  value,
  subtitle,
  icon,
  gradient = 'teal',
  index = 0,
}) => (
  <FadeIn delay={index * 0.1}>
    <motion.div
      whileHover={{ y: -4 }}
      transition={{ type: 'spring', stiffness: 300 }}
      className={cn(
        'glass rounded-2xl p-6 border-l-4 border-teal',
        'hover:shadow-lg hover:shadow-teal/20 transition-smooth'
      )}
    >
      <div className="flex items-start justify-between">
        <div className="flex-1">
          <p className="text-xs font-medium text-slate uppercase tracking-wider">
            {title}
          </p>
          <p className="text-3xl font-bold text-ink mt-3">{value}</p>
          {subtitle && <p className="text-xs text-slate mt-2">{subtitle}</p>}
        </div>
        {icon && <div className="text-2xl opacity-40 ml-2">{icon}</div>}
      </div>
    </motion.div>
  </FadeIn>
);

interface HeroSectionProps {
  title: string;
  description: string;
  badges?: string[];
}

export const HeroSection: React.FC<HeroSectionProps> = ({
  title,
  description,
  badges = [],
}) => (
  <FadeIn>
    <div className="glass rounded-2xl p-8 mb-8 border border-slate/20">
      <div className="flex flex-wrap gap-2 mb-4">
        {badges.map((badge) => (
          <span
            key={badge}
            className="inline-block px-3 py-1 text-xs font-medium text-sky bg-sky/20 border border-sky/30 rounded-full"
          >
            {badge}
          </span>
        ))}
      </div>
      <h1 className="text-4xl md:text-5xl font-bold text-ink mb-2 gradient-text">
        {title}
      </h1>
      <p className="text-slate max-w-2xl leading-relaxed">{description}</p>
    </div>
  </FadeIn>
);

interface ButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  href?: string;
  variant?: 'primary' | 'secondary' | 'outline';
  size?: 'sm' | 'md' | 'lg';
  loading?: boolean;
  children: ReactNode;
}

export const Button: React.FC<ButtonProps> = ({
  href,
  variant = 'primary',
  size = 'md',
  loading = false,
  children,
  className,
  ...props
}) => {
  const baseStyles =
    'font-medium rounded-lg transition-smooth flex items-center gap-2 justify-center';

  const variantStyles = {
    primary:
      'bg-gradient-to-r from-teal to-sky text-paper hover:shadow-lg hover:shadow-teal/40',
    secondary: 'bg-slate/20 text-ink border border-slate/30 hover:bg-slate/30',
    outline: 'border border-slate/30 text-ink hover:bg-slate/10',
  };

  const sizeStyles = {
    sm: 'px-3 py-1 text-sm',
    md: 'px-6 py-2 text-base',
    lg: 'px-8 py-3 text-lg',
  };

  const finalClassName = cn(
    baseStyles,
    variantStyles[variant],
    sizeStyles[size],
    loading && 'opacity-60 cursor-not-allowed',
    className
  );

  if (href) {
    return (
      <a href={href} className={finalClassName}>
        {children}
      </a>
    );
  }

  return (
    <motion.button
      whileHover={{ scale: 1.02 }}
      whileTap={{ scale: 0.98 }}
      disabled={loading}
      className={finalClassName}
      {...props}
    >
      {loading && <span className="animate-spin">⏳</span>}
      {children}
    </motion.button>
  );
};

interface CardProps {
  title?: string;
  children: ReactNode;
  className?: string;
}

export const Card: React.FC<CardProps> = ({
  title,
  children,
  className = '',
}) => (
  <div className={cn('glass rounded-xl p-6 border border-slate/20', className)}>
    {title && (
      <h3 className="text-lg font-semibold text-ink mb-4 flex items-center gap-2">
        <span className="h-1 w-1 bg-teal rounded-full" />
        {title}
      </h3>
    )}
    {children}
  </div>
);

interface LoadingSpinnerProps {
  size?: 'sm' | 'md' | 'lg';
  message?: string;
}

export const LoadingSpinner: React.FC<LoadingSpinnerProps> = ({
  size = 'md',
  message = 'Loading...',
}) => {
  const sizeMap = {
    sm: 'w-6 h-6',
    md: 'w-12 h-12',
    lg: 'w-16 h-16',
  };

  return (
    <div className="flex flex-col items-center justify-center gap-4">
      <motion.div
        animate={{ rotate: 360 }}
        transition={{ duration: 1, repeat: Infinity, ease: 'linear' }}
        className={cn(
          'border-2 border-slate/20 border-t-teal rounded-full',
          sizeMap[size]
        )}
      />
      <p className="text-slate">{message}</p>
    </div>
  );
};

interface AlertProps {
  type?: 'success' | 'error' | 'warning' | 'info';
  message: string;
  onClose?: () => void;
}

export const Alert: React.FC<AlertProps> = ({
  type = 'info',
  message,
  onClose,
}) => {
  const styles = {
    success: 'bg-emerald-500/20 border-emerald-500/40 text-emerald-100',
    error: 'bg-red-500/20 border-red-500/40 text-red-100',
    warning: 'bg-amber/20 border-amber/40 text-amber-100',
    info: 'bg-sky/20 border-sky/40 text-sky-100',
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: -8 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -8 }}
      className={cn(
        'border rounded-lg p-4 flex items-center justify-between',
        styles[type]
      )}
    >
      <p>{message}</p>
      {onClose && (
        <button
          onClick={onClose}
          className="ml-4 text-xl opacity-60 hover:opacity-100 transition-smooth"
        >
          ×
        </button>
      )}
    </motion.div>
  );
};
