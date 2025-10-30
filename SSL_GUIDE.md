# SSL Configuration Guide

This guide shows how to configure SSL/TLS for the SeamlessStreaming server.

## Quick Start (Development)

### Generate Self-Signed Certificates

```bash
# Make the script executable
chmod +x scripts/generate_ssl_cert.sh

# Generate certificates
./scripts/generate_ssl_cert.sh
```

### Start Server with SSL

```bash
# Start with self-signed certificates
cd backend
python3 streaming_server.py \
  --ssl-keyfile ../ssl_certs/server.key \
  --ssl-certfile ../ssl_certs/server.crt
```

### Access the Server

Open your browser to: `https://localhost:7860`

**Note**: You'll see a security warning for self-signed certificates. Click "Advanced" → "Proceed to localhost (unsafe)".

## Production Setup

### Option 1: Let's Encrypt (Recommended)

```bash
# Install certbot
sudo apt-get update
sudo apt-get install certbot

# Generate certificates (replace yourdomain.com)
sudo certbot certonly --standalone -d yourdomain.com

# Certificates will be in /etc/letsencrypt/live/yourdomain.com/
```

Start server with Let's Encrypt certificates:

```bash
python3 streaming_server.py \
  --host 0.0.0.0 \
  --port 443 \
  --ssl-keyfile /etc/letsencrypt/live/yourdomain.com/privkey.pem \
  --ssl-certfile /etc/letsencrypt/live/yourdomain.com/fullchain.pem
```

### Option 2: Custom CA Certificates

If you have certificates from a commercial CA:

```bash
python3 streaming_server.py \
  --ssl-keyfile /path/to/private.key \
  --ssl-certfile /path/to/certificate.crt \
  --ssl-ca-certs /path/to/ca-bundle.crt
```

## Command Line Options

| Option | Description | Example |
|--------|-------------|---------|
| `--host` | Host to bind to | `--host 0.0.0.0` |
| `--port` | Port to bind to | `--port 443` |
| `--ssl-keyfile` | Private key file | `--ssl-keyfile server.key` |
| `--ssl-certfile` | Certificate file | `--ssl-certfile server.crt` |
| `--ssl-ca-certs` | CA certificates | `--ssl-ca-certs ca-bundle.crt` |
| `--ssl-version` | TLS version | `--ssl-version TLSv1_2` |
| `--no-ssl-verify` | Disable cert verification | `--no-ssl-verify` |

## Frontend WebSocket Configuration

Update your frontend to use `wss://` instead of `ws://`:

```javascript
// For HTTPS sites, use secure WebSocket
const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
const wsUrl = `${protocol}//${window.location.host}/ws/stream`;
const websocket = new WebSocket(wsUrl);
```

## Security Best Practices

### 1. Certificate Management

```bash
# Set proper permissions for private keys
chmod 600 /path/to/private.key
chown root:root /path/to/private.key

# Certificates can be world-readable
chmod 644 /path/to/certificate.crt
```

### 2. Firewall Configuration

```bash
# Allow HTTPS traffic
sudo ufw allow 443/tcp

# If using custom port
sudo ufw allow 7860/tcp
```

### 3. Automatic Certificate Renewal

For Let's Encrypt, set up automatic renewal:

```bash
# Add to crontab
sudo crontab -e

# Add this line for automatic renewal
0 12 * * * /usr/bin/certbot renew --quiet --post-hook "systemctl reload nginx"
```

### 4. Reverse Proxy (Nginx)

For production, use Nginx as a reverse proxy:

```nginx
server {
    listen 443 ssl http2;
    server_name yourdomain.com;
    
    ssl_certificate /etc/letsencrypt/live/yourdomain.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/yourdomain.com/privkey.pem;
    
    # Modern SSL configuration
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-RSA-AES128-GCM-SHA256:ECDHE-RSA-AES256-GCM-SHA384;
    ssl_prefer_server_ciphers off;
    
    # WebSocket support
    location /ws/ {
        proxy_pass http://127.0.0.1:7860;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
    
    # Static files
    location / {
        proxy_pass http://127.0.0.1:7860;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}

# Redirect HTTP to HTTPS
server {
    listen 80;
    server_name yourdomain.com;
    return 301 https://$server_name$request_uri;
}
```

Then run your app on localhost:

```bash
python3 streaming_server.py --host 127.0.0.1 --port 7860
```

## Testing SSL Configuration

### 1. Check Certificate

```bash
# Test SSL certificate
openssl s_client -connect localhost:7860 -servername localhost

# Check certificate details
openssl x509 -in ssl_certs/server.crt -text -noout
```

### 2. Test WebSocket over SSL

```bash
# Install websocat for testing
curl --proto '=https' --tlsv1.2 -sSf https://websocat.pro/install.sh | sh

# Test WebSocket connection
echo '{"type":"start_stream","src_lang":"en","tgt_lang":"bn","sample_rate":16000}' | \
websocat wss://localhost:7860/ws/stream -k
```

### 3. Browser Testing

1. Open `https://localhost:7860`
2. Accept the security warning for self-signed certs
3. Test the microphone and translation

## Troubleshooting

### Common SSL Issues

1. **"SSL: CERTIFICATE_VERIFY_FAILED"**
   ```bash
   # Use --no-ssl-verify for development only
   python3 streaming_server.py --no-ssl-verify --ssl-keyfile ... --ssl-certfile ...
   ```

2. **"Permission denied" for certificate files**
   ```bash
   sudo chown $USER:$USER ssl_certs/*
   chmod 600 ssl_certs/server.key
   ```

3. **WebSocket fails over HTTPS**
   - Ensure using `wss://` not `ws://`
   - Check browser security settings
   - Verify certificate includes correct domains/IPs

4. **Port 443 already in use**
   ```bash
   # Find what's using port 443
   sudo netstat -tulpn | grep :443
   
   # Use different port
   python3 streaming_server.py --port 8443 --ssl-keyfile ... --ssl-certfile ...
   ```

## Performance Considerations

- **TLS 1.3**: Use `--ssl-version TLSv1_2` (TLS 1.3 not fully supported by uvicorn)
- **Cipher suites**: Modern browsers handle cipher selection automatically
- **Certificate chain**: Use full chain certificates for better compatibility
- **HSTS**: Consider enabling HTTP Strict Transport Security in production

With SSL enabled, your SeamlessStreaming server will have encrypted communication, which is essential for:
- ✅ Production deployments
- ✅ Microphone access in browsers (requires HTTPS)
- ✅ Data privacy and security
- ✅ Compliance with security policies