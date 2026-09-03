//! Minimal Standard MIDI File (SMF) reader/writer.
//! Minimal Standard MIDI File (SMF) reader/writer for the YufMusicGen client.

use anyhow::{Result, bail, ensure, Context};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Note {
    pub start: i32,
    pub duration: i32,
    pub pitch: u8,
    pub velocity: u8,
}

#[derive(Debug, Clone)]
pub struct Track {
    /// MIDI program number (0-127); -1 means the drum track.
    pub program: i32,
    pub is_drum: bool,
    pub name: String,
    pub notes: Vec<Note>,
}

#[derive(Debug, Clone)]
pub struct Score {
    pub ticks_per_quarter: u16,
    pub tracks: Vec<Track>,
}

impl Score {
    pub fn end_tick(&self) -> i32 {
        self.tracks
            .iter()
            .flat_map(|t| t.notes.iter().map(|n| n.start + n.duration))
            .max()
            .unwrap_or(0)
    }
}

// ---------------------------------------------------------------------------
// Reading
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct RawEvent {
    tick: i32,
    kind: RawEventKind,
}

#[derive(Debug, Clone)]
enum RawEventKind {
    NoteOn { channel: u8, pitch: u8, velocity: u8 },
    NoteOff { channel: u8, pitch: u8 },
    ProgramChange { channel: u8, program: u8 },
    TrackName(String),
    Other,
}

pub fn read_midi(path: &std::path::Path) -> Result<Score> {
    let bytes = std::fs::read(path)
        .with_context(|| format!("cannot read MIDI {}", path.display()))?;
    parse_midi(&bytes)
}

pub fn parse_midi(bytes: &[u8]) -> Result<Score> {
    let mut cursor = Cursor::new(bytes);
    ensure!(cursor.read_bytes(4)? == b"MThd", "not a MIDI file (missing MThd)");
    let header_len = cursor.read_u32()? as usize;
    let _format = cursor.read_u16()?;
    let track_count = cursor.read_u16()?;
    let division = cursor.read_u16()?;
    cursor.skip(header_len - 6)?;
    ensure!(division & 0x8000 == 0, "SMPTE timing is not supported");

    let mut raw_tracks: Vec<Vec<RawEvent>> = Vec::new();
    for _ in 0..track_count {
        ensure!(
            cursor.read_bytes(4)? == b"MTrk",
            "expected MTrk chunk while parsing MIDI"
        );
        let chunk_len = cursor.read_u32()? as usize;
        let start = cursor.pos;
        let mut events = Vec::new();
        let mut tick = 0i32;
        let mut running_status: Option<u8> = None;
        while cursor.pos < start + chunk_len {
            tick += cursor.read_variable_length()? as i32;
            let status = cursor.peek()?;
            let (status, data_len) = if status & 0x80 != 0 {
                cursor.advance(1)?;
                (status, 0)
            } else {
                let status = running_status
                    .with_context(|| "running status used before any status byte")?;
                (status, 1)
            };
            running_status = Some(status);
            match status & 0xF0 {
                0x80 => {
                    let channel = status & 0x0F;
                    let pitch = cursor.read_byte()?;
                    let _velocity = cursor.read_byte()?;
                    events.push(RawEvent {
                        tick,
                        kind: RawEventKind::NoteOff { channel, pitch },
                    });
                }
                0x90 => {
                    let channel = status & 0x0F;
                    let pitch = cursor.read_byte()?;
                    let velocity = cursor.read_byte()?;
                    if velocity == 0 {
                        events.push(RawEvent {
                            tick,
                            kind: RawEventKind::NoteOff { channel, pitch },
                        });
                    } else {
                        events.push(RawEvent {
                            tick,
                            kind: RawEventKind::NoteOn {
                                channel,
                                pitch,
                                velocity,
                            },
                        });
                    }
                }
                0xC0 => {
                    let channel = status & 0x0F;
                    let program = cursor.read_byte()?;
                    events.push(RawEvent {
                        tick,
                        kind: RawEventKind::ProgramChange { channel, program },
                    });
                }
                0xF0 => {
                    // System exclusive / meta.
                    let status = status;
                    if status == 0xFF {
                        let meta_type = cursor.read_byte()?;
                        let length = cursor.read_variable_length()? as usize;
                        let payload = cursor.read_bytes(length)?.to_vec();
                        if meta_type == 0x03 {
                            let name = String::from_utf8_lossy(&payload).into_owned();
                            events.push(RawEvent {
                                tick,
                                kind: RawEventKind::TrackName(name),
                            });
                        } else {
                            events.push(RawEvent {
                                tick,
                                kind: RawEventKind::Other,
                            });
                        }
                    } else {
                        let length = cursor.read_variable_length()? as usize;
                        cursor.skip(length)?;
                        events.push(RawEvent {
                            tick,
                            kind: RawEventKind::Other,
                        });
                    }
                }
                _ => {
                    // 0xA0 (poly aftertouch), 0xB0 (CC), 0xD0 (channel pressure),
                    // 0xE0 (pitch bend): two data bytes.
                    let _ = cursor.read_byte()?;
                    let _ = cursor.read_byte()?;
                    let _ = data_len;
                    events.push(RawEvent {
                        tick,
                        kind: RawEventKind::Other,
                    });
                }
            }
        }
        cursor.pos = start + chunk_len;
        raw_tracks.push(events);
    }

    build_score(division, raw_tracks)
}

fn build_score(division: u16, raw_tracks: Vec<Vec<RawEvent>>) -> Result<Score> {
    let mut tracks: Vec<Track> = Vec::new();
    for events in &raw_tracks {
        // Group by channel: each channel becomes one track.  Channel 9 is the
        // GM drum channel.
        let mut by_channel: Vec<(u8, Vec<RawEvent>)> = Vec::new();
        let mut program_of_channel = [0u8; 16];
        for event in events {
            match &event.kind {
                RawEventKind::ProgramChange { channel, program } => {
                    program_of_channel[*channel as usize] = *program;
                }
                _ => {}
            }
        }
        for event in events {
            let channel = match &event.kind {
                RawEventKind::NoteOn { channel, .. }
                | RawEventKind::NoteOff { channel, .. }
                | RawEventKind::ProgramChange { channel, .. } => Some(*channel),
                RawEventKind::TrackName(_) => None,
                RawEventKind::Other => None,
            };
            let Some(channel) = channel else { continue };
            match by_channel.iter_mut().find(|(c, _)| *c == channel) {
                Some((_, list)) => list.push(event.clone()),
                None => by_channel.push((channel, vec![event.clone()])),
            }
        }
        let mut name = String::new();
        for event in events {
            if let RawEventKind::TrackName(value) = &event.kind {
                name = value.clone();
                break;
            }
        }
        for (channel, events) in by_channel {
            let is_drum = channel == 9;
            let program = program_of_channel[channel as usize] as i32;
            let notes = note_events_to_notes(&events, is_drum)?;
            if notes.is_empty() {
                continue;
            }
            tracks.push(Track {
                program,
                is_drum,
                name: name.clone(),
                notes,
            });
        }
    }
    Ok(Score {
        ticks_per_quarter: division,
        tracks,
    })
}

fn note_events_to_notes(events: &[RawEvent], _is_drum: bool) -> Result<Vec<Note>> {
    let mut open: Vec<(i32, u8, u8)> = Vec::new(); // (start, pitch, velocity)
    let mut notes: Vec<Note> = Vec::new();
    for event in events {
        match &event.kind {
            RawEventKind::NoteOn {
                channel: _,
                pitch,
                velocity,
            } => {
                open.push((event.tick, *pitch, *velocity));
            }
            RawEventKind::NoteOff { channel: _, pitch } => {
                if let Some(index) = open.iter().rposition(|(_, p, _)| p == pitch) {
                    let (start, p, velocity) = open.remove(index);
                    let duration = (event.tick - start).max(1);
                    notes.push(Note {
                        start,
                        duration,
                        pitch: p,
                        velocity,
                    });
                }
            }
            _ => {}
        }
    }
    // Close any notes that never received a NoteOff at the end of the track.
    let end = events.last().map(|e| e.tick).unwrap_or(0);
    for (start, pitch, velocity) in open {
        notes.push(Note {
            start,
            duration: (end - start).max(1),
            pitch,
            velocity,
        });
    }
    notes.sort_by_key(|n| (n.start, n.duration, n.pitch));
    Ok(notes)
}

struct Cursor<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> Cursor<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, pos: 0 }
    }
    fn peek(&self) -> Result<u8> {
        self.bytes
            .get(self.pos)
            .copied()
            .with_context(|| "unexpected end of MIDI data")
    }
    fn advance(&mut self, n: usize) -> Result<()> {
        ensure!(self.pos + n <= self.bytes.len(), "MIDI data truncated");
        self.pos += n;
        Ok(())
    }
    fn read_byte(&mut self) -> Result<u8> {
        let value = self.peek()?;
        self.pos += 1;
        Ok(value)
    }
    fn read_bytes(&mut self, n: usize) -> Result<&'a [u8]> {
        ensure!(self.pos + n <= self.bytes.len(), "MIDI data truncated");
        let slice = &self.bytes[self.pos..self.pos + n];
        self.pos += n;
        Ok(slice)
    }
    fn skip(&mut self, n: usize) -> Result<()> {
        self.advance(n)
    }
    fn read_u16(&mut self) -> Result<u16> {
        let bytes = self.read_bytes(2)?;
        Ok(u16::from_be_bytes([bytes[0], bytes[1]]))
    }
    fn read_u32(&mut self) -> Result<u32> {
        let bytes = self.read_bytes(4)?;
        Ok(u32::from_be_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
    }
    fn read_variable_length(&mut self) -> Result<u32> {
        let mut value: u32 = 0;
        for _ in 0..4 {
            let byte = self.read_byte()?;
            value = (value << 7) | (byte & 0x7F) as u32;
            if byte & 0x80 == 0 {
                return Ok(value);
            }
        }
        bail!("invalid variable-length quantity in MIDI")
    }
}

// ---------------------------------------------------------------------------
// Writing
// ---------------------------------------------------------------------------

pub fn write_midi(path: &std::path::Path, score: &Score) -> Result<()> {
    let bytes = render_midi(score)?;
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)?;
        }
    }
    std::fs::write(path, &bytes)
        .with_context(|| format!("cannot write MIDI {}", path.display()))
}

pub fn render_midi(score: &Score) -> Result<Vec<u8>> {
    let mut out = Vec::new();
    out.extend_from_slice(b"MThd");
    out.extend_from_slice(&6u32.to_be_bytes());
    out.extend_from_slice(&1u16.to_be_bytes()); // format 1
    out.extend_from_slice(&(score.tracks.len() as u16).to_be_bytes());
    out.extend_from_slice(&score.ticks_per_quarter.to_be_bytes());

    let tpq = score.ticks_per_quarter as i64;
    for (_track_index, track) in score.tracks.iter().enumerate() {
        let mut track_bytes = Vec::new();
        let channel = if track.is_drum { 9 } else { (track.program % 16) as u8 };
        // Track name meta.
        let name = if track.name.is_empty() {
            format!(
                "{}",
                if track.is_drum {
                    "Drums"
                } else {
                    crate::instruments::gm_name(track.program)
                }
            )
        } else {
            track.name.clone()
        };
        push_meta(&mut track_bytes, 0, 0x03, name.as_bytes());
        // Program change at tick 0 (drums get program 0).
        let program = if track.is_drum { 0 } else { track.program as u8 };
        push_channel_event(&mut track_bytes, 0, 0xC0, channel, program);

        let mut sorted: Vec<&Note> = track.notes.iter().collect();
        sorted.sort_by_key(|n| (n.start, n.pitch, n.duration));
        let mut last_tick = 0i64;
        for note in &sorted {
            let start = note.start as i64 * tpq / score.ticks_per_quarter as i64;
            let end = (note.start as i64 + note.duration as i64) * tpq
                / score.ticks_per_quarter as i64;
            let start = start.max(last_tick);
            let end = end.max(start + 1);
            push_variable_length(&mut track_bytes, (start - last_tick) as u64);
            track_bytes.push(0x90 | channel);
            track_bytes.push(note.pitch);
            track_bytes.push(note.velocity);
            push_variable_length(&mut track_bytes, (end - start) as u64);
            track_bytes.push(0x80 | channel);
            track_bytes.push(note.pitch);
            track_bytes.push(0x00);
            last_tick = end;
        }
        push_meta(&mut track_bytes, 0, 0x2F, &[]);

        out.extend_from_slice(b"MTrk");
        out.extend_from_slice(&(track_bytes.len() as u32).to_be_bytes());
        out.extend_from_slice(&track_bytes);
    }
    Ok(out)
}

fn push_variable_length(out: &mut Vec<u8>, mut value: u64) {
    let mut buffer = [0u8; 4];
    let mut index = 3;
    buffer[index] = (value & 0x7F) as u8;
    value >>= 7;
    while value > 0 {
        index -= 1;
        buffer[index] = ((value & 0x7F) as u8) | 0x80;
        value >>= 7;
    }
    out.extend_from_slice(&buffer[index..]);
}

fn push_meta(out: &mut Vec<u8>, delta: u64, meta_type: u8, payload: &[u8]) {
    push_variable_length(out, delta);
    out.push(0xFF);
    out.push(meta_type);
    push_variable_length(out, payload.len() as u64);
    out.extend_from_slice(payload);
}

fn push_channel_event(out: &mut Vec<u8>, delta: u64, status: u8, channel: u8, data: u8) {
    push_variable_length(out, delta);
    out.push((status & 0xF0) | (channel & 0x0F));
    out.push(data);
}
