#!/bin/bash

# Quick SSL Certificate Fix Script
# Regenerates certificates with proper extensions for web servers

echo "🔧 Fixing SSL certificate compatibility issues..."

CERT_DIR="ssl_certs"
DOMAIN="localhost"
KEY_FILE="server.key"
CERT_FILE="server.crt"
DAYS=365

# Create directory if it doesn't exist
mkdir -p "$CERT_DIR"
cd "$CERT_DIR"

# Remove old certificates
echo "🗑️ Removing old certificates..."
rm -f "$KEY_FILE" "$CERT_FILE" server.csr server.conf

# Generate new private key with proper algorithm
echo "🔑 Generating RSA private key..."
openssl genrsa -out "$KEY_FILE" 2048

# Create OpenSSL config with proper extensions
echo "📝 Creating SSL configuration..."
cat > server.conf <<EOF
[req]
distinguished_name = req_distinguished_name
req_extensions = v3_req
prompt = no
default_bits = 2048

[req_distinguished_name]
C = US
ST = State
L = City
O = SeamlessStreaming
OU = Development
CN = localhost
emailAddress = admin@localhost

[v3_req]
basicConstraints = CA:FALSE
keyUsage = critical, digitalSignature, keyEncipherment, keyAgreement
extendedKeyUsage = critical, serverAuth, clientAuth
subjectAltName = @alt_names
nsCertType = server
nsComment = "OpenSSL Generated Server Certificate"

[alt_names]
DNS.1 = localhost
DNS.2 = *.localhost
DNS.3 = 127.0.0.1
IP.1 = 127.0.0.1
IP.2 = ::1
EOF

# Generate certificate signing request
echo "📄 Creating certificate signing request..."
openssl req -new -key "$KEY_FILE" -out server.csr -config server.conf

# Generate self-signed certificate with v3 extensions
echo "📜 Generating self-signed certificate with proper extensions..."
openssl x509 -req -in server.csr -signkey "$KEY_FILE" -out "$CERT_FILE" \
    -days "$DAYS" -extensions v3_req -extfile server.conf

# Verify certificate
echo "🔍 Verifying certificate..."
openssl x509 -in "$CERT_FILE" -text -noout | grep -A 10 "X509v3 extensions:"

# Set proper permissions
chmod 600 "$KEY_FILE"
chmod 644 "$CERT_FILE"

# Clean up
rm server.csr server.conf

echo ""
echo "✅ SSL certificates regenerated successfully!"
echo ""
echo "📁 Certificate files:"
echo "   Private key: $(pwd)/$KEY_FILE (permissions: 600)"
echo "   Certificate: $(pwd)/$CERT_FILE (permissions: 644)"
echo ""
echo "🔍 Certificate details:"
openssl x509 -in "$CERT_FILE" -noout -subject -issuer -dates

echo ""
echo "🚀 Start your server with:"
echo "   python3 backend/streaming_server.py --ssl-keyfile $(pwd)/$KEY_FILE --ssl-certfile $(pwd)/$CERT_FILE"
echo ""
echo "🌐 Access at: https://localhost:7860"
echo ""
echo "⚠️ If you still get SSL errors, try:"
echo "   1. Clear browser cache and cookies for localhost"
echo "   2. Try in incognito/private mode"
echo "   3. Use Chrome with --ignore-certificate-errors flag (development only)"