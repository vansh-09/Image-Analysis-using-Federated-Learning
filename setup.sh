#!/bin/bash

# MediSync FL - React Migration Setup Script
# This script sets up both frontend and backend

set -e

echo "🚀 MediSync FL - React Migration Setup"
echo "========================================"

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Backend setup
echo -e "\n${BLUE}📦 Setting up Backend...${NC}"
cd backend

if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

echo "Activating virtual environment..."
source venv/bin/activate

echo "Installing Python dependencies..."
pip install -r requirements.txt

echo -e "${GREEN}✓ Backend setup complete${NC}"

# Frontend setup
echo -e "\n${BLUE}⚛️  Setting up Frontend...${NC}"
cd ../frontend

if [ ! -d "node_modules" ]; then
    echo "Installing npm dependencies..."
    npm install
else
    echo "Dependencies already installed, skipping npm install"
fi

echo -e "${GREEN}✓ Frontend setup complete${NC}"

# Print instructions
echo -e "\n${BLUE}📋 Setup Complete!${NC}"
echo ""
echo "To start developing:"
echo ""
echo "1. Start Backend (Terminal 1):"
echo "   cd backend"
echo "   source venv/bin/activate"
echo "   python app.py"
echo ""
echo "2. Start Frontend (Terminal 2):"
echo "   cd frontend"
echo "   npm run dev"
echo ""
echo "3. Open http://localhost:3000 in your browser"
echo ""
echo -e "${GREEN}Happy coding! 🎉${NC}"
