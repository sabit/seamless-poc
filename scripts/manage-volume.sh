#!/bin/bash

# Docker volume management script for SeamlessStreaming Translation Service

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

VOLUME_NAME="seamless-model-cache"

print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

show_help() {
    echo "Docker Volume Management for SeamlessStreaming Translation"
    echo ""
    echo "Usage: $0 [COMMAND]"
    echo ""
    echo "Commands:"
    echo "  create     Create the model cache volume"
    echo "  inspect    Show volume details and contents"
    echo "  backup     Backup the volume to a tar file"
    echo "  restore    Restore volume from a tar file"
    echo "  clean      Remove the volume (WARNING: deletes cached models)"
    echo "  size       Show volume size and disk usage"
    echo "  list       List all Docker volumes"
    echo "  help       Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 create"
    echo "  $0 backup model-cache-backup.tar"
    echo "  $0 restore model-cache-backup.tar"
    echo ""
}

create_volume() {
    print_status "Creating Docker volume '$VOLUME_NAME'..."
    if docker volume create $VOLUME_NAME >/dev/null 2>&1; then
        print_success "Volume '$VOLUME_NAME' created successfully"
    else
        print_warning "Volume '$VOLUME_NAME' already exists"
    fi
}

inspect_volume() {
    print_status "Inspecting volume '$VOLUME_NAME'..."
    
    if ! docker volume inspect $VOLUME_NAME >/dev/null 2>&1; then
        print_error "Volume '$VOLUME_NAME' does not exist"
        return 1
    fi
    
    echo ""
    echo "📦 Volume Details:"
    docker volume inspect $VOLUME_NAME
    
    echo ""
    print_status "Volume contents:"
    docker run --rm -v $VOLUME_NAME:/data alpine ls -la /data 2>/dev/null || print_warning "Could not list volume contents"
}

backup_volume() {
    local backup_file=${1:-"seamless-model-cache-$(date +%Y%m%d_%H%M%S).tar"}
    
    if ! docker volume inspect $VOLUME_NAME >/dev/null 2>&1; then
        print_error "Volume '$VOLUME_NAME' does not exist"
        return 1
    fi
    
    print_status "Backing up volume '$VOLUME_NAME' to '$backup_file'..."
    
    docker run --rm \
        -v $VOLUME_NAME:/data \
        -v "$(pwd):/backup" \
        alpine tar czf "/backup/$backup_file" -C /data .
    
    if [ $? -eq 0 ]; then
        print_success "Backup completed: $backup_file"
        ls -lh "$backup_file" 2>/dev/null || true
    else
        print_error "Backup failed"
        return 1
    fi
}

restore_volume() {
    local backup_file="$1"
    
    if [ -z "$backup_file" ]; then
        print_error "Please specify backup file to restore from"
        echo "Usage: $0 restore <backup_file.tar>"
        return 1
    fi
    
    if [ ! -f "$backup_file" ]; then
        print_error "Backup file '$backup_file' not found"
        return 1
    fi
    
    print_status "Restoring volume '$VOLUME_NAME' from '$backup_file'..."
    
    # Create volume if it doesn't exist
    docker volume create $VOLUME_NAME >/dev/null 2>&1
    
    docker run --rm \
        -v $VOLUME_NAME:/data \
        -v "$(pwd):/backup" \
        alpine sh -c "cd /data && tar xzf /backup/$backup_file"
    
    if [ $? -eq 0 ]; then
        print_success "Restore completed from: $backup_file"
    else
        print_error "Restore failed"
        return 1
    fi
}

clean_volume() {
    print_warning "⚠️  This will permanently delete the cached models!"
    print_warning "You will need to re-download the SeamlessM4T model (~2.5GB) on next startup."
    echo ""
    read -p "Are you sure you want to delete volume '$VOLUME_NAME'? (y/N): " -n 1 -r
    echo
    
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        if docker volume rm $VOLUME_NAME 2>/dev/null; then
            print_success "Volume '$VOLUME_NAME' removed successfully"
        else
            print_error "Failed to remove volume (may be in use by running container)"
            print_status "Stop the container first: docker stop seamless-translator-app"
        fi
    else
        print_status "Volume removal cancelled"
    fi
}

show_size() {
    print_status "Docker system disk usage:"
    docker system df -v
    
    echo ""
    if docker volume inspect $VOLUME_NAME >/dev/null 2>&1; then
        print_status "Volume '$VOLUME_NAME' details:"
        docker volume inspect $VOLUME_NAME --format "{{.Mountpoint}}" | xargs sudo du -sh 2>/dev/null || {
            print_warning "Cannot determine volume size (requires root access)"
            print_status "Volume exists and is managed by Docker"
        }
    else
        print_warning "Volume '$VOLUME_NAME' does not exist"
    fi
}

list_volumes() {
    print_status "All Docker volumes:"
    docker volume ls
    
    echo ""
    print_status "Seamless-related volumes:"
    docker volume ls --filter name=seamless
}

# Main command handling
case "${1:-help}" in
    create)
        create_volume
        ;;
    inspect)
        inspect_volume
        ;;
    backup)
        backup_volume "$2"
        ;;
    restore)
        restore_volume "$2"
        ;;
    clean)
        clean_volume
        ;;
    size)
        show_size
        ;;
    list)
        list_volumes
        ;;
    help|--help|-h)
        show_help
        ;;
    *)
        print_error "Unknown command: $1"
        echo ""
        show_help
        exit 1
        ;;
esac