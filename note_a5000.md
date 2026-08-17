# A5000 server access — SSH notes (issue, fix, key location)

Notes for connecting to the shared A5000 research box. Keep this handy; it explains the SSH
hang we hit and how to get into the box from any PC.

## Server

| | |
|---|---|
| Host (IP) | `220.69.209.182` |
| User | `pheng` |
| Hostname | `uc-Precision-7920-Tower` (Ubuntu 22.04, 3× RTX A5000, 64 cores, 192 GB RAM) |
| Project dir | `~/data/kcr/structure-kcr` |
| Data dir | `/mnt/DATA_1/pheng/kcr` (only big disk writable to `pheng`; DATA_2/DATA_3 are root-only) |

This is a **shared** server (login banner rules): academic/research use only, do **not** use root,
do **not** kill other users' processes, do **not** power it off, and **check `nvidia-smi` before
using the GPUs**.

---

## The issue: `ssh` hangs then times out

Symptom — plain `ssh pheng@220.69.209.182` stalls and eventually:
```
ssh_dispatch_run_fatal: Connection to 220.69.209.182 port 22: Connection timed out
```
With `ssh -v` it always froze at:
```
debug1: expecting SSH2_MSG_KEX_ECDH_REPLY      <-- hangs here
```
`ping` worked fine (the box was reachable), and it happened even when the server CPU was idle.

### Root cause
A **network path-MTU / middlebox problem**, not the server. OpenSSH's default `KEXINIT`
handshake packet is large (it advertises many algorithms). On some network paths — typically
after **switching WiFi/network or toggling a VPN** (a VPN lowers the MTU) — a router/middlebox
silently **drops that large packet**, so the key exchange never completes and ssh hangs.

> Note: a *separate* problem the same day was overloading the box (running the render with 56
> Chromium workers), which pegged all CPUs and made even a good SSH path time out. Fix for that:
> keep render workers modest (≈16–24). See below. The MTU issue is the one this note is about.

### The fix
Pin a **single small kex + cipher** so the handshake packet stays under the MTU. This is already
baked into `~/.ssh/config` (see below) and also gives passwordless login via a key.

---

## SSH key — location & how to copy to another PC

A key was generated so VS Code Remote-SSH and plain `ssh` connect with **no password**
(VS Code opens several connections and password auth kept hitting its 17 s timeout).

### Key files on THIS PC
```
Private key : C:\Users\USER\.ssh\id_ed25519        <-- SECRET. copy this to other PCs. never share publicly.
Public key  : C:\Users\USER\.ssh\id_ed25519.pub    <-- safe to share; this is what lives on the server
```
The public key is installed in the server's `~/.ssh/authorized_keys`.

### To access from ANOTHER PC
1. **Copy the private key** `id_ed25519` to the new PC's `.ssh` folder:
   - Windows: `C:\Users\<you>\.ssh\id_ed25519`
   - macOS/Linux: `~/.ssh/id_ed25519` then `chmod 600 ~/.ssh/id_ed25519`
   (Copy it over a safe channel — USB, password manager, encrypted transfer. It's a secret.)

2. **Add the SSH config block** to that PC's `~/.ssh/config` (create the file if missing). This is
   what makes plain `ssh pheng@220.69.209.182` work past the MTU hang:
   ```sshconfig
   Host 220.69.209.182
     HostName 220.69.209.182
     User pheng
     IdentityFile ~/.ssh/id_ed25519
     # Path-MTU workaround: pin a small kex + cipher so the handshake packet
     # stays under the MTU (fixes the "expecting SSH2_MSG_KEX_ECDH_REPLY" hang).
     KexAlgorithms curve25519-sha256,curve25519-sha256@libssh.org
     Ciphers chacha20-poly1305@openssh.com
     ServerAliveInterval 30
     ServerAliveCountMax 6
     TCPKeepAlive yes
   ```
   (On Windows use the full path `C:\Users\<you>\.ssh\id_ed25519` for `IdentityFile`.)

3. **Test:** `ssh pheng@220.69.209.182 "echo OK; hostname"` → should print `OK` and the hostname
   with no password prompt.

### If you ever need to re-create the key from scratch
```powershell
ssh-keygen -t ed25519 -f $env:USERPROFILE\.ssh\id_ed25519 -N '""'
type $env:USERPROFILE\.ssh\id_ed25519.pub | ssh pheng@220.69.209.182 "mkdir -p ~/.ssh && cat >> ~/.ssh/authorized_keys && chmod 700 ~/.ssh && chmod 600 ~/.ssh/authorized_keys && echo KEY_INSTALLED"
```

### Emergency: if the box is CPU-pegged and SSH won't complete the handshake
Force the small cipher and retry in a loop until one attempt lands, which runs the kill:
```powershell
while ($true) { ssh -o ConnectTimeout=8 pheng@220.69.209.182 "pkill -9 -f build_dataset.py; pkill -9 -f chrome; echo KILLED"; if ($LASTEXITCODE -eq 0) { 'KILLED'; break }; 'retry'; Start-Sleep 8 }
```

---

## Running the render/training (reminders)

- **Always use tmux** so a disconnect can't kill a multi-hour run:
  ```bash
  tmux new -s render          # start
  # Ctrl-b then d  to detach ; reattach later:
  tmux attach -t render
  ```
- **Keep CPU workers modest (≈16–24).** 56 froze the whole box and locked everyone out.
  ```bash
  cd ~/data/kcr/structure-kcr
  nvidia-smi                                       # shared box: confirm GPUs are free first
  WORKERS=24 ./run_v4_a5000.sh 2>&1 | tee v4_a5000.log
  ```
- **All big data lives on `/mnt/DATA_1/pheng/kcr`** (root `/` is 97 % full — never write big data there).
- Full pipeline runbook: `LINUX_A5000_RUNBOOK.md`.
