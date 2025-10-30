#!/bin/bash

# SSL Certificate Generation Script for SeamlessStreaming Server
# This script generates self-signed SSL certificates for development use

set -e

CERT_DIR="ssl_certs"
DOMAIN="localhost"
KEY_FILE="server.key"
CERT_FILE="server.crt"
DAYS=365

echo "🔒 Generating SSL certificates for SeamlessStreaming server..."

# Create certificate directory
mkdir -p "$CERT_DIR"
cd "$CERT_DIR"

# Generate private key
echo "🔑 Generating private key..."
openssl genrsa -out "$KEY_FILE" 2048

# Generate certificate signing request
echo "📝 Creating certificate signing request..."
openssl req -new -key "$KEY_FILE" -out server.csr -subj "/C=US/ST=State/L=City/O=Organization/OU=OrgUnit/CN=$DOMAIN"

# Generate self-signed certificate
echo "📜 Generating self-signed certificate..."
openssl x509 -req -days "$DAYS" -in server.csr -signkey "$KEY_FILE" -out "$CERT_FILE"

# Create certificate with proper extensions for web servers
echo "🌐 Creating certificate with proper server extensions..."
cat > server.conf <<EOF
[req]
distinguished_name = req_distinguished_name
req_extensions = v3_req
prompt = no

[req_distinguished_name]
C = US
ST = State
L = City
O = Organization
OU = OrgUnit
CN = $DOMAIN

[v3_req]
basicConstraints = CA:FALSE
keyUsage = critical, digitalSignature, keyEncipherment, keyAgreement
extendedKeyUsage = critical, serverAuth
subjectAltName = @alt_names

[alt_names]
DNS.1 = localhost
DNS.2 = *.localhost
IP.1 = 127.0.0.1
IP.2 = ::1
EOF

# Generate new certificate with SAN
openssl req -new -x509 -key "$KEY_FILE" -out "$CERT_FILE" -days "$DAYS" -config server.conf -extensions v3_req

# Clean up
rm server.csr server.conf

echo "✅ SSL certificates generated successfully!"
echo ""
echo "📁 Certificate files:"
echo "   Private key: $(pwd)/$KEY_FILE"
echo "   Certificate: $(pwd)/$CERT_FILE"
echo ""
echo "🚀 To start the server with SSL:"
echo "   python3 streaming_server.py --ssl-keyfile $(pwd)/$KEY_FILE --ssl-certfile $(pwd)/$CERT_FILE"
echo ""
echo "🌐 Access your server at: https://localhost:7860"
echo ""
echo "⚠️  Note: Browsers will show security warnings for self-signed certificates."
echo "   Click 'Advanced' -> 'Proceed to localhost (unsafe)' to continue."
echo ""
echo "🔧 For production, use certificates from a trusted CA like Let's Encrypt:"
echo "   certbot certonly --standalone -d yourdomain.com"