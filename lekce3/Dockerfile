FROM langchain/langgraphjs-api:20

WORKDIR /deps/lekce3
COPY . .

RUN ls -la
RUN pwd
RUN if [ -f package.json ]; then echo "package.json exists" && cat package.json; else echo "package.json not found"; fi
RUN if [ -f package.json ]; then npm install; fi 