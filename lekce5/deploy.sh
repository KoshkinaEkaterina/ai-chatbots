#!/bin/bash

# Install Render CLI if needed
curl https://render.com/download/cli | bash

# Login to Render
render login

# Create new web service
render create webservice \
  --name "twilio-search-bot" \
  --repo "https://github.com/yourusername/your-repo" \
  --branch "main" \
  --build-command "pip install -r requirements.txt" \
  --start-command "uvicorn twilio_bot:app --host 0.0.0.0 --port \$PORT" \
  --env "OPENAI_API_KEY=$OPENAI_API_KEY" \
  --env "TWILIO_ACCOUNT_SID=$TWILIO_ACCOUNT_SID" \
  --env "TWILIO_AUTH_TOKEN=$TWILIO_AUTH_TOKEN" \
  --env "TAVILY_API_KEY=$TAVILY_API_KEY" 