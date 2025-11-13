FROM node:20-alpine

WORKDIR /app

COPY package.json ./
COPY pnpm-lock.yaml* ./
RUN corepack enable \
    && if [ -f pnpm-lock.yaml ]; then \
         pnpm install --frozen-lockfile; \
       else \
         pnpm install; \
       fi

COPY . .

# Build Next.js app during Docker build (faster than runtime)
RUN pnpm build

EXPOSE 3000

CMD ["pnpm", "start", "-p", "3000"]
