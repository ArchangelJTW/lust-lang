use std::io::{self, Read, Write};
use std::sync::{Arc, Mutex};

/// VirtualConsole: Buffering and interception of standard I/O streams (stdin, stdout, stderr).
/// Completely isolates outputs under simulation to prevent host terminal noise and race conditions.
#[derive(Debug, Clone)]
pub struct VirtualConsole {
    stdin_buffer: Arc<Mutex<Vec<u8>>>,
    stdin_offset: Arc<Mutex<usize>>,
    stdout_buffer: Arc<Mutex<Vec<u8>>>,
    stderr_buffer: Arc<Mutex<Vec<u8>>>,
    captured_only: bool,
}

impl Default for VirtualConsole {
    fn default() -> Self {
        Self::new()
    }
}

impl VirtualConsole {
    pub fn new() -> Self {
        Self {
            stdin_buffer: Arc::new(Mutex::new(Vec::new())),
            stdin_offset: Arc::new(Mutex::new(0)),
            stdout_buffer: Arc::new(Mutex::new(Vec::new())),
            stderr_buffer: Arc::new(Mutex::new(Vec::new())),
            captured_only: false,
        }
    }

    pub fn captured() -> Self {
        Self {
            stdin_buffer: Arc::new(Mutex::new(Vec::new())),
            stdin_offset: Arc::new(Mutex::new(0)),
            stdout_buffer: Arc::new(Mutex::new(Vec::new())),
            stderr_buffer: Arc::new(Mutex::new(Vec::new())),
            captured_only: true,
        }
    }

    pub fn is_captured(&self) -> bool {
        self.captured_only || crate::sim::is_simulated_mode()
    }

    pub fn write_stdout(&self, data: &[u8]) {
        self.stdout_buffer.lock().unwrap().extend_from_slice(data);
        if !self.is_captured() {
            let mut out = io::stdout();
            let _ = out.write_all(data);
            let _ = out.flush();
        }
    }

    pub fn write_stderr(&self, data: &[u8]) {
        self.stderr_buffer.lock().unwrap().extend_from_slice(data);
        if !self.is_captured() {
            let mut err = io::stderr();
            let _ = err.write_all(data);
            let _ = err.flush();
        }
    }

    pub fn set_stdin(&self, data: impl AsRef<[u8]>) {
        let mut buf = self.stdin_buffer.lock().unwrap();
        buf.clear();
        buf.extend_from_slice(data.as_ref());
        *self.stdin_offset.lock().unwrap() = 0;
    }

    pub fn read_stdin(&self, buf: &mut [u8]) -> io::Result<usize> {
        let stdin = self.stdin_buffer.lock().unwrap();
        let mut offset = self.stdin_offset.lock().unwrap();
        if *offset < stdin.len() {
            let available = &stdin[*offset..];
            let n = available.len().min(buf.len());
            buf[..n].copy_from_slice(&available[..n]);
            *offset += n;
            Ok(n)
        } else if self.is_captured() {
            Ok(0)
        } else {
            drop(stdin);
            drop(offset);
            io::stdin().read(buf)
        }
    }

    pub fn read_line_stdin(&self) -> io::Result<String> {
        let stdin = self.stdin_buffer.lock().unwrap();
        let mut offset = self.stdin_offset.lock().unwrap();
        if *offset < stdin.len() {
            let available = &stdin[*offset..];
            if let Some(newline_pos) = available.iter().position(|&b| b == b'\n') {
                let line_bytes = &available[..=newline_pos];
                let line = String::from_utf8_lossy(line_bytes).into_owned();
                *offset += newline_pos + 1;
                Ok(line)
            } else {
                let line = String::from_utf8_lossy(available).into_owned();
                *offset += available.len();
                Ok(line)
            }
        } else if self.is_captured() {
            Ok(String::new())
        } else {
            drop(stdin);
            drop(offset);
            let mut line = String::new();
            io::stdin().read_line(&mut line)?;
            Ok(line)
        }
    }

    pub fn stdout_bytes(&self) -> Vec<u8> {
        self.stdout_buffer.lock().unwrap().clone()
    }

    pub fn stderr_bytes(&self) -> Vec<u8> {
        self.stderr_buffer.lock().unwrap().clone()
    }

    pub fn stdout_string(&self) -> String {
        String::from_utf8_lossy(&self.stdout_buffer.lock().unwrap()).into_owned()
    }

    pub fn stderr_string(&self) -> String {
        String::from_utf8_lossy(&self.stderr_buffer.lock().unwrap()).into_owned()
    }

    pub fn clear_stdout(&self) {
        self.stdout_buffer.lock().unwrap().clear();
    }

    pub fn clear_stderr(&self) {
        self.stderr_buffer.lock().unwrap().clear();
    }

    pub fn clear(&self) {
        self.clear_stdout();
        self.clear_stderr();
        self.stdin_buffer.lock().unwrap().clear();
        *self.stdin_offset.lock().unwrap() = 0;
    }
}
