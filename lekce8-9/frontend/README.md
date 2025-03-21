# Product Recommendation Chat Frontend

A React-based frontend for the product recommendation chatbot. Features real-time chat, product cards, and dynamic recommendations.

## 🚀 Tech Stack
- Next.js 14
- React
- TypeScript
- Tailwind CSS
- Shadcn/ui Components

## 🛠️ Local Development

### Prerequisites
- Node.js 18+ 
- npm or yarn
- Backend API running (see ../README.md)

### Installation
```bash
# Install dependencies
npm install
# or
yarn install

# Create .env.local file
cp .env.example .env.local
```

### Environment Variables
Create a `.env.local` file with:
```env
NEXT_PUBLIC_API_URL=http://localhost:8001  # Local API
# For production:
# NEXT_PUBLIC_API_URL=https://your-api-url.com
```

### Running Locally
```bash
# Development mode
npm run dev
# or
yarn dev

# Build for production
npm run build
# or
yarn build

# Start production build
npm start
# or
yarn start
```

Visit http://localhost:3000 to see the app.

## 🧪 Testing

### Unit Tests
```bash
# Run unit tests
npm test
# or
yarn test

# Watch mode
npm test:watch
# or
yarn test:watch
```

### E2E Tests
```bash
# Run Cypress tests
npm run cypress
# or
yarn cypress

# Open Cypress UI
npm run cypress:open
# or
yarn cypress:open
```

## 📱 Features
- Real-time chat interface
- Product card display
- Markdown message rendering
- Responsive design
- Loading states
- Error handling
- Product comparison view

## 🚀 Deployment to Vercel

### Method 1: Using Vercel CLI

1. Install Vercel CLI:
```bash
npm i -g vercel
```

2. Login to Vercel:
```bash
vercel login
```

3. Deploy:
```bash
# Deploy to preview
vercel

# Deploy to production
vercel --prod
```

### Method 2: Using Vercel Dashboard

1. Push your code to GitHub

2. Go to [Vercel Dashboard](https://vercel.com/dashboard)

3. Click "New Project"

4. Import your GitHub repository

5. Configure project:
   - Framework Preset: Next.js
   - Root Directory: `ai-chatbots/lekce8-9/frontend`
   - Environment Variables: Add `NEXT_PUBLIC_API_URL`

6. Click "Deploy"

### Environment Variables on Vercel
Make sure to add these in your Vercel project settings: 