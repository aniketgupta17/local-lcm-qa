import React, { useState, useEffect, useRef, useCallback } from 'react';
import {
  AppShell,
  Navbar,
  Header,
  Group,
  Text,
  ActionIcon,
  ColorSchemeProvider,
  type ColorScheme,
  MantineProvider,
  Button,
  ScrollArea,
  SegmentedControl,
  Container,
  Paper,
  TextInput,
  Textarea,
  Divider,
  Loader,
  Avatar,
  Tooltip,
  Badge,
  Title,
  Tabs,
  Notification,
  Box,
  Popover,
  Card,
  Transition,
} from '@mantine/core';
import { Sun, MoonStars, Upload, FileText, Send, Trash, Refresh, Download, InfoCircle, ChevronRight } from 'tabler-icons-react';
import axios from 'axios';
import {
  BarChart, Bar, XAxis, YAxis, Tooltip as RechartsTooltip, ResponsiveContainer, PieChart, Pie, Cell, 
  ScatterChart, Scatter, ZAxis, Legend, CartesianGrid
} from 'recharts';

const BACKEND_URL = 'http://localhost:8001';

// Type definitions
interface DocumentInfo {
  id: string;
  filename: string;
  status: string;
  processed_at?: string;
  chunk_count?: number;
}

// New type for chats
interface ChatSession {
  id: string;
  name: string;
  documentIds: string[];
  createdAt: string;
}

interface ChunkInfo {
  chunk_index: number;
  text: string;
  summary: string;
  metadata?: any;
  excerpt?: string;
  word_count?: number;
  char_count?: number;
}

// Define a type for docDetails
interface DocDetails {
  chunks: ChunkInfo[];
  file_info?: { filename?: string };
  filename?: string;
  [key: string]: unknown;
}

// Theme colors for light/dark
const themeColors = {
  light: {
    primary: '#4361ee',
    secondary: '#f3f4f6',
    accent: '#e0e7ff',
    background: '#f8fafc',
    card: '#fff',
    text: '#22223b',
    border: '#eaeaea',
    hover: '#f1f5f9',
    gradient: 'linear-gradient(90deg, #4361ee 0%, #3a0ca3 100%)',
    fontFamily: 'Nunito, sans-serif',
    headingFont: 'Montserrat, sans-serif',
  },
  dark: {
    primary: '#4cc9f0',
    secondary: '#232946',
    accent: '#232946',
    background: '#0f172a',
    card: '#1e293b',
    text: '#e0e0e0',
    border: '#334155',
    hover: '#232946',
    gradient: 'linear-gradient(90deg, #4cc9f0 0%, #4361ee 100%)',
    fontFamily: 'Nunito, sans-serif',
    headingFont: 'Montserrat, sans-serif',
  },
};

// Add Google Fonts import for Nunito and Montserrat
const fontLink = document.createElement('link');
fontLink.href = 'https://fonts.googleapis.com/css2?family=Nunito:wght@400;600;700&family=Montserrat:wght@500;600;700&display=swap';
fontLink.rel = 'stylesheet';
document.head.appendChild(fontLink);

function Sidebar({ theme, toggleTheme, documents, onSelectDocument, onUploadClick, selectedId, chatSessions, selectedChatId, onSelectChat, onCreateChat }: {
  theme: ColorScheme;
  toggleTheme: () => void;
  documents: DocumentInfo[];
  onSelectDocument: (id: string) => void;
  onUploadClick: () => void;
  selectedId: string | null;
  chatSessions: ChatSession[];
  selectedChatId: string | null;
  onSelectChat: (id: string) => void;
  onCreateChat: () => void;
}) {
  const colors = themeColors[theme];
  const [searchTerm, setSearchTerm] = useState('');
  const [activeTab, setActiveTab] = useState<'chats' | 'documents'>('chats');
  const [editingChatName, setEditingChatName] = useState<string | null>(null);
  const [newChatName, setNewChatName] = useState('');

  // Filter documents based on search term
  const filteredDocs = documents.filter(doc =>
    doc.filename?.toLowerCase().includes(searchTerm.toLowerCase())
  );

  // Filter chats based on search term
  const filteredChats = chatSessions.filter(chat =>
    chat.name?.toLowerCase().includes(searchTerm.toLowerCase())
  );

  return (
    <Navbar 
      width={{ base: 300 }} 
      p="md" 
      style={{ 
        borderRight: `1px solid ${colors.border}`,
        background: theme === 'dark' ? colors.background : colors.card,
      }}
    >
      <Navbar.Section mb="md">
        <Group position="apart">
          <Text 
            weight={700} 
            size="lg" 
            style={{ 
              letterSpacing: 0.5, 
              fontFamily: colors.headingFont,
              background: colors.gradient,
              WebkitBackgroundClip: 'text',
              WebkitTextFillColor: 'transparent',
            }}
          >
            Scientific QA
          </Text>
          <Tooltip label="New Chat" withArrow position="right">
            <ActionIcon variant="light" color="blue" size="lg" onClick={onCreateChat} radius="md">
              <Send size={20} />
            </ActionIcon>
          </Tooltip>
        </Group>
      </Navbar.Section>

      <Navbar.Section mb="md">
        <Tabs 
          value={activeTab} 
          onTabChange={(value) => setActiveTab(value as 'chats' | 'documents')}
          radius="md"
        >
          <Tabs.List grow>
            <Tabs.Tab value="chats">Chats</Tabs.Tab>
            <Tabs.Tab value="documents">Documents</Tabs.Tab>
          </Tabs.List>
        </Tabs>
      </Navbar.Section>

      <Navbar.Section>
        <TextInput
          placeholder={activeTab === 'chats' ? "Search chats..." : "Search documents..."}
          value={searchTerm}
          onChange={e => setSearchTerm(e.currentTarget.value)}
          mb="md"
          icon={activeTab === 'chats' ? <Send size={16} /> : <FileText size={16} />}
          radius="md"
        />
        <Group position="apart" mb="sm">
          <Text 
            weight={600} 
            size="sm" 
            style={{ fontFamily: colors.headingFont }}
          >
            {activeTab === 'chats' ? 'Chat Sessions' : 'Document Library'}
          </Text>
          <Tooltip label={activeTab === 'chats' ? "New Chat" : "Upload Document"} withArrow position="right">
            <ActionIcon 
              variant="light" 
              color="blue" 
              size="sm" 
              onClick={activeTab === 'chats' ? onCreateChat : onUploadClick} 
              radius="md"
            >
              {activeTab === 'chats' ? <Send size={16} /> : <Upload size={16} />}
            </ActionIcon>
          </Tooltip>
        </Group>
        <ScrollArea style={{ height: 'calc(100vh - 320px)' }} mt="sm">
          {activeTab === 'chats' ? (
            filteredChats.length === 0 ? (
              <Text color="dimmed" size="sm" align="center" mt="md">No chats yet</Text>
            ) : (
              filteredChats.map((chat) => (
                <Paper
                  key={chat.id}
                  shadow="xs"
                  p="md"
                  mb="sm"
                  radius="md"
                  style={{
                    borderLeft: selectedChatId === chat.id ? `4px solid ${colors.primary}` : 'none',
                    backgroundColor: selectedChatId === chat.id ? 
                      (theme === 'dark' ? '#2d3748' : '#f0f5ff') : 
                      (theme === 'dark' ? colors.card : colors.card),
                    cursor: 'pointer',
                    transition: 'all 0.2s ease',
                  }}
                  onClick={() => onSelectChat(chat.id)}
                >
                  <Group position="apart">
                    <Group>
                      <Send 
                        size={22} 
                        color={selectedChatId === chat.id ? colors.primary : undefined} 
                      />
                      <div style={{ overflow: 'hidden' }}>
                        <Text 
                          weight={600}
                          style={{ 
                            whiteSpace: 'nowrap', 
                            overflow: 'hidden', 
                            textOverflow: 'ellipsis',
                            fontFamily: colors.fontFamily,
                            color: selectedChatId === chat.id ? colors.primary : undefined
                          }}
                        >
                          {chat.name || 'Unnamed Chat'}
                        </Text>
                        <Text size="xs" color="dimmed">
                          {chat.documentIds.length} document{chat.documentIds.length !== 1 ? 's' : ''}
                        </Text>
                      </div>
                    </Group>
                    
                    {selectedChatId === chat.id && (
                      <ActionIcon 
                        size="sm" 
                        variant="subtle"
                        onClick={(e: React.MouseEvent) => {
                          e.stopPropagation();
                          setEditingChatName(chat.id);
                          setNewChatName(chat.name);
                        }}
                      >
                        <InfoCircle size={16} />
                      </ActionIcon>
                    )}
                  </Group>
                </Paper>
              ))
            )
          ) : (
            filteredDocs.length === 0 ? (
              <Text color="dimmed" size="sm" align="center" mt="md">No documents yet</Text>
            ) : (
              filteredDocs.map((doc) => (
                <Paper
                  key={doc.id}
                  shadow="xs"
                  p="md"
                  mb="sm"
                  radius="md"
                  style={{
                    borderLeft: selectedId === doc.id ? `4px solid ${colors.primary}` : 'none',
                    backgroundColor: selectedId === doc.id ? 
                      (theme === 'dark' ? '#2d3748' : '#f0f5ff') : 
                      (theme === 'dark' ? colors.card : colors.card),
                    cursor: 'pointer',
                    transition: 'all 0.2s ease',
                  }}
                  onClick={() => onSelectDocument(doc.id)}
                >
                  <Group>
                    <FileText 
                      size={22} 
                      color={selectedId === doc.id ? colors.primary : undefined} 
                    />
                    <div style={{ overflow: 'hidden' }}>
                      <Text 
                        weight={600}
                        style={{ 
                          whiteSpace: 'nowrap', 
                          overflow: 'hidden', 
                          textOverflow: 'ellipsis',
                          fontFamily: colors.fontFamily,
                          color: selectedId === doc.id ? colors.primary : undefined
                        }}
                      >
                        {doc.filename || 'Unnamed'}
                      </Text>
                      <Text size="xs" color="dimmed">
                        {doc.chunk_count || '?'} chunks • {doc.status}
                      </Text>
                    </div>
                  </Group>
                </Paper>
              ))
            )
          )}
        </ScrollArea>
      </Navbar.Section>
      
      <Navbar.Section>
        <Text size="xs" color="dimmed" align="center" style={{ fontStyle: 'italic', marginTop: 'auto', paddingTop: 20 }}>
          All processing happens locally. Your data never leaves your machine.
        </Text>
      </Navbar.Section>
    </Navbar>
  );
}

function AppHeader({ health, theme, toggleTheme }: { health: boolean; theme: ColorScheme; toggleTheme: () => void }) {
  const colors = themeColors[theme];
  return (
    <Header 
      height={70} 
      p="xs" 
      style={{ 
        background: theme === 'dark' ? colors.background : 'white', 
        borderBottom: `1px solid ${colors.border}` 
      }}
    >
      <Group position="apart" align="center" style={{ height: '100%' }}>
        <Group>
          <Title 
            order={2} 
            style={{ 
              fontFamily: colors.headingFont, 
              fontWeight: 700, 
              letterSpacing: 0.5,
              background: colors.gradient,
              WebkitBackgroundClip: 'text',
              WebkitTextFillColor: 'transparent',
            }}
          >
            Scientific Document QA
          </Title>
        </Group>
        <Group spacing="md">
          <Group spacing={6}>
            <ActionIcon 
              variant="light" 
              color={theme === 'dark' ? 'blue' : 'indigo'} 
              onClick={toggleTheme}
              size="lg"
              radius="md"
            >
              {theme === 'dark' ? <Sun size={20} /> : <MoonStars size={20} />}
            </ActionIcon>
            <Text size="sm" color="dimmed">
              {theme === 'dark' ? 'Light Mode' : 'Dark Mode'}
            </Text>
          </Group>
          <Badge 
            color={health ? 'teal' : 'red'} 
            size="lg" 
            variant="filled" 
            radius="md"
            style={{ paddingLeft: 15, paddingRight: 15 }}
          >
            {health ? 'Backend Online' : 'Backend Offline'}
          </Badge>
        </Group>
      </Group>
    </Header>
  );
}

function DocumentInput({ theme, onUpload, uploading, error }: {
  theme: ColorScheme;
  onUpload: (file: File | null, text: string) => void;
  uploading: boolean;
  error: string | null;
}) {
  const colors = themeColors[theme];
  const [uploadMethod, setUploadMethod] = useState<'file' | 'text'>('file');
  const [textContent, setTextContent] = useState('');
  const [file, setFile] = useState<File | null>(null);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setFile(e.target.files[0]);
    }
  };

  const handleProcess = () => {
    onUpload(uploadMethod === 'file' ? file : null, uploadMethod === 'text' ? textContent : '');
  };

  return (
    <Paper 
      shadow="sm" 
      p="lg" 
      mb="md" 
      radius="lg"
      style={{ 
        borderLeft: `4px solid ${colors.primary}`,
        backgroundColor: theme === 'dark' ? colors.card : colors.card,
      }}
    >
      <Text 
        weight={700} 
        size="lg" 
        mb="lg" 
        style={{ 
          fontFamily: colors.headingFont, 
          fontSize: 22,
          color: theme === 'dark' ? colors.text : colors.text,
        }}
      >
        Document Input
      </Text>
      <Tabs 
        value={uploadMethod} 
        onTabChange={v => setUploadMethod(v as 'file' | 'text')}
        radius="md"
        styles={{
          tabLabel: {
            fontFamily: colors.fontFamily,
            fontWeight: 600,
          },
        }}
      >
        <Tabs.List>
          <Tabs.Tab value="file">Upload File</Tabs.Tab>
          <Tabs.Tab value="text">Paste Text</Tabs.Tab>
        </Tabs.List>
        <Tabs.Panel value="file" pt="md">
          <Paper 
            p="md" 
            withBorder 
            style={{ 
              borderStyle: 'dashed', 
              cursor: 'pointer',
              borderColor: theme === 'dark' ? colors.border : colors.border,
              backgroundColor: theme === 'dark' ? 'rgba(255,255,255,0.03)' : 'rgba(0,0,0,0.01)',
            }}
          >
            <input 
              type="file" 
              accept=".pdf,.docx,.txt,.html,.csv,.json" 
              onChange={handleFileChange} 
              style={{ display: 'none' }} 
              id="file-upload"
            />
            <label htmlFor="file-upload" style={{ cursor: 'pointer', width: '100%', display: 'block' }}>
              <Group position="center" spacing="xs">
                <Upload size={30} color={colors.primary} />
                <Text align="center">Click or drag files here</Text>
                <Text size="xs" color="dimmed" align="center">
                  Supports PDF, DOCX, TXT, HTML, CSV, JSON
                </Text>
              </Group>
            </label>
          </Paper>
          {file && (
            <Paper p="sm" mt="md" radius="md" withBorder>
              <Group>
                <FileText size={16} />
                <Text size="sm">Selected: {file.name}</Text>
              </Group>
            </Paper>
          )}
        </Tabs.Panel>
        <Tabs.Panel value="text" pt="md">
          <Textarea
            placeholder="Paste your text here..."
            value={textContent}
            onChange={e => setTextContent(e.currentTarget.value)}
            minRows={6}
            radius="md"
            style={{
              fontFamily: colors.fontFamily,
            }}
          />
        </Tabs.Panel>
      </Tabs>
      <Button 
        mt="lg" 
        color={theme === 'dark' ? 'cyan' : 'blue'} 
        onClick={handleProcess} 
        loading={uploading} 
        disabled={uploading || (uploadMethod === 'file' && !file) || (uploadMethod === 'text' && !textContent)}
        fullWidth
        size="md"
        radius="md"
        style={{
          fontFamily: colors.fontFamily,
          fontWeight: 600,
        }}
      >
        Process Document
      </Button>
      {error && <Notification color="red" mt="md" withCloseButton={false}>{error}</Notification>}
    </Paper>
  );
}

export default function App() {
  const [theme, setTheme] = useState<ColorScheme>('dark');
  const [health, setHealth] = useState(false);
  const [documents, setDocuments] = useState<DocumentInfo[]>([]);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [docDetails, setDocDetails] = useState<DocDetails | null>(null);
  const [summary, setSummary] = useState('');
  const [loadingDoc, setLoadingDoc] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [uploadError, setUploadError] = useState<string | null>(null);
  
  // New states for chat functionality
  const [showUpload, setShowUpload] = useState(true);
  const [chatSessions, setChatSessions] = useState<ChatSession[]>([]);
  const [selectedChatId, setSelectedChatId] = useState<string | null>(null);
  const [chatQAMap, setChatQAMap] = useState<Record<string, { question: string; answer: string; sources?: any[] }[]>>({});
  const [addingDocToChat, setAddingDocToChat] = useState(false);
  const [combinedChunkCount, setCombinedChunkCount] = useState(0);
  const [combinedDocDetails, setCombinedDocDetails] = useState<DocDetails | null>(null);
  const [isUpdatingCombined, setIsUpdatingCombined] = useState(false);
  const [docDetailsMap, setDocDetailsMap] = useState<Record<string, DocDetails>>({});
  const [forceUpdate, setForceUpdate] = useState(0); // Add a state to force rerender
  const [showCombinedView, setShowCombinedView] = useState(false);

  const toggleTheme = () => setTheme((prev) => (prev === 'dark' ? 'light' : 'dark'));

  // Create a new chat session
  const createNewChat = () => {
    const newChatId = `chat-${Date.now()}`;
    const newChat: ChatSession = {
      id: newChatId,
      name: `Chat ${chatSessions.length + 1}`,
      documentIds: [],
      createdAt: new Date().toISOString(),
    };
    
    setChatSessions(prev => [...prev, newChat]);
    setSelectedChatId(newChatId);
    setSelectedId(null); // Clear selected document
    setDocDetails(null);
    setCombinedDocDetails(null);
    setSummary('');
    setChatQAMap(prev => ({...prev, [newChatId]: []}));
    setShowUpload(true); // Show upload for new chat
    setCombinedChunkCount(0);
  };

  // Add document to chat
  const addDocumentToChat = (chatId: string, docId: string) => {
    setChatSessions(prev => 
      prev.map(chat => 
        chat.id === chatId 
          ? {...chat, documentIds: [...chat.documentIds.filter(id => id !== docId), docId]} 
          : chat
      )
    );
  };

  // Get chat documents - memoized to prevent unnecessary re-renders
  const getChatDocuments = useCallback((chatId: string | null) => {
    if (!chatId) return [];
    const chat = chatSessions.find(c => c.id === chatId);
    if (!chat) return [];
    return documents.filter(doc => chat.documentIds.includes(doc.id));
  }, [chatSessions, documents]);

  // Select chat
  const handleSelectChat = (chatId: string) => {
    setSelectedChatId(chatId);
    
    // If chat has documents, select the first one by default
    const chat = chatSessions.find(c => c.id === chatId);
    if (chat?.documentIds.length) {
      const firstDocId = chat.documentIds[0];
      setSelectedId(firstDocId);
    } else {
      setSelectedId(null);
      setDocDetails(null);
      setCombinedDocDetails(null);
      setSummary('');
    }
    
    // If chat has no documents, show upload
    setShowUpload(chat?.documentIds.length === 0);
    setAddingDocToChat(false);
    
    // Reset combined chunk count
    setCombinedChunkCount(0);
  };

  // Fetch document details and store in cache
  const fetchDocDetails = useCallback(async (docId: string) => {
    // Check if we already have the details cached
    if (docDetailsMap[docId]) {
      return docDetailsMap[docId];
    }
    
    try {
      const res = await axios.get(`${BACKEND_URL}/api/documents/${docId}`);
      const details = res.data;
      
      // Cache the result
      setDocDetailsMap(prev => ({
        ...prev,
        [docId]: details
      }));
      
      return details;
    } catch (error) {
      console.error("Error fetching document details:", error);
      return null;
    }
  }, [docDetailsMap]);

  // Combine document details for visualization
  const updateCombinedDocDetails = useCallback(async () => {
    if (!selectedChatId || isUpdatingCombined) return;
    
    const chatDocs = getChatDocuments(selectedChatId);
    if (chatDocs.length <= 0) {
      setCombinedDocDetails(null);
      setShowCombinedView(false);
      return;
    }
    
    if (chatDocs.length === 1) {
      // For a single document, just use its details directly
      if (docDetails && selectedId === chatDocs[0].id) {
        setCombinedDocDetails(docDetails);
        setShowCombinedView(false);
        return;
      }
      
      // If the document details aren't loaded yet, fetch them
      const details = await fetchDocDetails(chatDocs[0].id);
      if (details) {
        setCombinedDocDetails(details);
        setShowCombinedView(false);
      }
      return;
    }
    
    // For multiple documents, we need to combine them
    setIsUpdatingCombined(true);
    setShowCombinedView(true);
    
    try {
      // Get details for all documents in the chat
      let totalChunks: ChunkInfo[] = [];
      let chunkIndex = 0;
      
      // Use Promise.all to fetch all documents in parallel
      const docDetailsArray = await Promise.all(
        chatDocs.map(doc => fetchDocDetails(doc.id))
      );
      
      // Combine all chunks from all documents
      docDetailsArray.forEach((details, docIndex) => {
        if (details && details.chunks) {
          const adjustedChunks = details.chunks.map((chunk: ChunkInfo) => ({
            ...chunk,
            chunk_index: chunkIndex++,
            docName: chatDocs[docIndex].filename || `Document ${docIndex + 1}`,
            docId: chatDocs[docIndex].id
          }));
          
          totalChunks = [...totalChunks, ...adjustedChunks];
        }
      });
      
      // Create combined doc details
      const combined: DocDetails = {
        chunks: totalChunks,
        filename: `Combined (${chatDocs.length} documents)`,
        file_info: {
          filename: `Combined (${chatDocs.length} documents)`
        }
      };
      
      setCombinedDocDetails(combined);
      setCombinedChunkCount(totalChunks.length);
    } catch (error) {
      console.error("Error combining document details:", error);
    } finally {
      setIsUpdatingCombined(false);
    }
  }, [selectedChatId, docDetails, selectedId, getChatDocuments, fetchDocDetails, isUpdatingCombined]);
  
  // Update combined details when chat or selected document changes
  useEffect(() => {
    if (selectedChatId) {
      const chatDocs = getChatDocuments(selectedChatId);
      if (chatDocs.length > 0) {
        updateCombinedDocDetails();
      }
    }
  }, [selectedChatId, getChatDocuments, updateCombinedDocDetails, forceUpdate]);

  // Fetch document details when selected
  useEffect(() => {
    if (!selectedId) return;
    
    setLoadingDoc(true);
    fetchDocDetails(selectedId)
      .then(details => {
        if (details) {
          setDocDetails(details);
          
          // If this is the only document in the chat, set it as combined doc details too
          if (!selectedChatId || getChatDocuments(selectedChatId).length <= 1) {
            setCombinedDocDetails(details);
            setCombinedChunkCount((details.chunks || []).length);
            setShowCombinedView(false);
          } else {
            // If there are multiple documents, update the combined view
            updateCombinedDocDetails();
          }
          
          // Fetch summary
          axios.get(`${BACKEND_URL}/api/document/${selectedId}/download`)
            .then(r => {
              setSummary(r.data.markdown || '');
            })
            .catch(() => setSummary(''));
        }
      })
      .catch(() => {
        setDocDetails(null);
        setCombinedDocDetails(null);
      })
      .finally(() => setLoadingDoc(false));
  }, [selectedId, selectedChatId, getChatDocuments, fetchDocDetails, forceUpdate, updateCombinedDocDetails]);

  // Health check
  useEffect(() => {
    axios.get(`${BACKEND_URL}/api/health`).then(res => setHealth(res.status === 200)).catch(() => setHealth(false));
  }, []);

  // Fetch document list
  const fetchDocuments = () => {
    axios.get(`${BACKEND_URL}/api/documents`).then(res => {
      setDocuments(res.data.documents || []);
    });
  };
  
  useEffect(() => { fetchDocuments(); }, []);

  // Upload handler
  const handleUpload = async (file: File | null, text: string) => {
    setUploading(true);
    setUploadError(null);
    try {
      let res;
      if (file) {
        const formData = new FormData();
        formData.append('file', file);
        res = await axios.post(`${BACKEND_URL}/api/upload`, formData, { headers: { 'Content-Type': 'multipart/form-data' } });
      } else if (text) {
        res = await axios.post(`${BACKEND_URL}/api/upload`, { text });
      } else {
        setUploadError('Please select a file or enter text.');
        setUploading(false);
        return;
      }
      if (res.data.document_id) {
        const newDocId = res.data.document_id;
        
        // If adding to an existing chat
        if (selectedChatId && (addingDocToChat || chatSessions.find(c => c.id === selectedChatId)?.documentIds.length === 0)) {
          addDocumentToChat(selectedChatId, newDocId);
          setAddingDocToChat(false);
        } 
        // If no chat selected, create a new one
        else if (!selectedChatId) {
          createNewChat();
          // Get the ID of the chat we just created
          const newChatId = `chat-${Date.now() - 1}`; // Approximate ID since we don't have a return value
          addDocumentToChat(newChatId, newDocId);
          setSelectedChatId(newChatId);
        }
        
        fetchDocuments();
        
        // Set the selected ID and trigger a fetch
        setSelectedId(newDocId);
        
        // Force the document to reload after a delay to ensure backend processing is complete
        setTimeout(() => {
          // Clear the document details to force a refetch
          setDocDetails(null);
          setCombinedDocDetails(null);
          
          // Force update to trigger the useEffect
          setForceUpdate(prev => prev + 1);
          
          // Hide upload after a successful document load
          setShowUpload(false);
          
          // If this is a second document, automatically show combined view
          const chatDocs = getChatDocuments(selectedChatId);
          if (chatDocs && chatDocs.length > 1) {
            setShowCombinedView(true);
          }
        }, 1000);
      }
    } catch (err: any) {
      setUploadError(err?.response?.data?.error || 'Failed to upload/process document.');
    } finally {
      setUploading(false);
    }
  };

  // Download summary handler
  const handleDownload = async () => {
    if (!selectedId) return;
    try {
      const res = await axios.get(`${BACKEND_URL}/api/document/${selectedId}/download`);
      const blob = new Blob([res.data.markdown || JSON.stringify(res.data, null, 2)], { type: 'text/markdown' });
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = res.data.filename || 'document_summary.md';
      a.click();
      window.URL.revokeObjectURL(url);
    } catch {
      // ignore
    }
  };

  return (
    <MantineProvider 
      theme={{ 
        colorScheme: theme, 
        fontFamily: themeColors[theme].fontFamily,
        headings: { fontFamily: themeColors[theme].headingFont },
        colors: {
          dark: [
            '#C1C2C5',
            '#A6A7AB',
            '#909296',
            '#5c5f66',
            '#373A40',
            '#2C2E33',
            '#25262b',
            '#1A1B1E',
            '#141517',
            '#101113',
          ],
        },
      }} 
      withGlobalStyles 
      withNormalizeCSS
    >
      <ColorSchemeProvider colorScheme={theme} toggleColorScheme={toggleTheme}>
        <AppShell
          padding="md"
          navbar={
            <Sidebar 
              theme={theme} 
              toggleTheme={toggleTheme} 
              documents={documents} 
              onSelectDocument={(id) => {
                setSelectedId(id);
                setShowUpload(false); // Always hide upload when selecting a document
              }} 
              onUploadClick={() => {
                setShowUpload(true);
                setSelectedId(null);
                setDocDetails(null);
                setSummary('');
                setAddingDocToChat(selectedChatId !== null);
              }}
              selectedId={selectedId}
              chatSessions={chatSessions}
              selectedChatId={selectedChatId}
              onSelectChat={handleSelectChat}
              onCreateChat={createNewChat}
            />
          }
          header={<AppHeader health={health} theme={theme} toggleTheme={toggleTheme} />}
          styles={(theme) => ({
            main: {
              backgroundColor: themeColors[theme.colorScheme].background,
              padding: '16px',
              overflowY: 'auto',
            },
          })}
        >
          <Container size="xl" py="md" style={{ minHeight: 'calc(100vh - 80px)' }}>
            {/* Add Document FAB button - always visible when a chat is selected */}
            {selectedChatId && !showUpload && (
              <ActionIcon
                variant="filled"
                color="blue"
                radius="xl"
                size="xl"
                style={{
                  position: 'fixed',
                  bottom: '30px',
                  right: '30px',
                  zIndex: 1000,
                  boxShadow: '0 4px 12px rgba(0,0,0,0.1)'
                }}
                onClick={() => {
                  setShowUpload(true);
                  setAddingDocToChat(true);
                }}
              >
                <Upload size={24} />
              </ActionIcon>
            )}
            
            {selectedChatId && (
              <Box mb="xl" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <Text weight={700} size="xl" style={{ 
                  fontFamily: themeColors[theme].headingFont,
                  color: theme === 'dark' ? themeColors[theme].text : themeColors[theme].text,
                }}>
                  {chatSessions.find(c => c.id === selectedChatId)?.name || 'Chat'}
                </Text>
                <Group spacing="md">
                  {/* Primary add document button - always visible when in a chat */}
                  <Button 
                    leftIcon={<Upload size={16} />} 
                    variant="filled"
                    color="blue"
                    size="md"
                    radius="md"
                    onClick={() => {
                      setShowUpload(true);
                      setAddingDocToChat(true);
                    }}
                  >
                    Add Documents
                  </Button>

                  {/* Toggle between combined and individual view */}
                  {selectedChatId && getChatDocuments(selectedChatId).length > 1 && !showUpload && (
                    <Button
                      variant="light"
                      color={theme === 'dark' ? 'cyan' : 'blue'}
                      onClick={() => setShowCombinedView(!showCombinedView)}
                    >
                      {showCombinedView ? 'Show Individual' : 'Show Combined'}
                    </Button>
                  )}
                  
                  {/* Button to toggle between visualization and LLM interface when in narrow layout */}
                  {getChatDocuments(selectedChatId).length > 0 && (
                    <ActionIcon 
                      variant="light"
                      color={theme === 'dark' ? 'cyan' : 'blue'}
                      size="lg"
                      radius="md"
                      onClick={() => setShowUpload(!showUpload)}
                    >
                      {showUpload ? <Send size={20} /> : <Upload size={20} />}
                    </ActionIcon>
                  )}
                </Group>
              </Box>
            )}
            
            {/* Document list for current chat - Moved up and improved */}
            {selectedChatId && getChatDocuments(selectedChatId).length > 0 && (
              <Paper 
                shadow="sm" 
                p="md" 
                mb="xl" 
                radius="lg"
                style={{ 
                  borderLeft: `4px solid ${themeColors[theme].primary}`,
                  backgroundColor: theme === 'dark' ? themeColors[theme].card : themeColors[theme].card,
                }}
              >
                <Group position="apart" mb="md">
                  <Text weight={600} size="md" style={{ fontFamily: themeColors[theme].headingFont }}>
                    Documents in this chat
                  </Text>
                  <Button 
                    variant="light" 
                    leftIcon={<Upload size={14} />}
                    size="sm"
                    onClick={() => {
                      setShowUpload(true);
                      setAddingDocToChat(true);
                    }}
                  >
                    Add More Documents
                  </Button>
                </Group>
                
                <Group spacing="md">
                  {getChatDocuments(selectedChatId).map(doc => (
                    <Badge 
                      key={doc.id} 
                      size="lg" 
                      radius="md" 
                      style={{ 
                        cursor: 'pointer',
                        padding: '10px 15px',
                        borderColor: selectedId === doc.id ? themeColors[theme].primary : undefined,
                        borderWidth: selectedId === doc.id ? '2px' : '1px',
                        borderStyle: 'solid',
                        backgroundColor: selectedId === doc.id ? 
                          (theme === 'dark' ? 'rgba(76, 201, 240, 0.1)' : 'rgba(67, 97, 238, 0.1)') : 
                          undefined
                      }}
                      onClick={() => {
                        setSelectedId(doc.id);
                        setShowCombinedView(false); // Show individual view when clicking a specific document
                      }}
                    >
                      <Group spacing={6}>
                        <FileText size={16} />
                        <Text size="sm" weight={600}>
                          {doc.filename || 'Unnamed'}
                        </Text>
                      </Group>
                    </Badge>
                  ))}
                </Group>
              </Paper>
            )}
            
            <Box style={{ 
              display: 'grid', 
              gridTemplateColumns: showUpload ? '1fr' : 'minmax(0, 1fr) minmax(0, 1fr)',
              gridGap: '24px',
              height: 'calc(100vh - 180px)',
              position: 'relative',
            }}>
              {/* Upload section only shown when needed */}
              {showUpload && (
                <Box style={{ 
                  display: 'flex', 
                  flexDirection: 'column',
                  height: '100%',
                  overflowY: 'auto',
                  position: 'relative',
                }}>
                  <Text 
                    weight={700} 
                    size="xl" 
                    style={{ 
                      fontFamily: themeColors[theme].headingFont,
                      color: theme === 'dark' ? themeColors[theme].text : themeColors[theme].text,
                      marginBottom: '20px',
                      paddingTop: '20px',
                    }}
                  >
                    {addingDocToChat ? `Add Document to ${chatSessions.find(c => c.id === selectedChatId)?.name || 'Chat'}` : 'Upload Document'}
                  </Text>
                  
                  {/* Go back button when adding to an existing chat */}
                  {addingDocToChat && selectedChatId && (
                    <Button
                      variant="subtle"
                      leftIcon={<ChevronRight size={16} style={{ transform: 'rotate(180deg)' }} />}
                      onClick={() => {
                        setShowUpload(false);
                        setAddingDocToChat(false);
                      }}
                      style={{ position: 'absolute', top: 20, right: 0 }}
                    >
                      Back to Chat
                    </Button>
                  )}
                  
                  <DocumentInput 
                    theme={theme} 
                    onUpload={handleUpload} 
                    uploading={uploading} 
                    error={uploadError} 
                  />

                  {!selectedChatId && !selectedId && (
                    <Paper 
                      shadow="sm" 
                      p="xl" 
                      radius="lg"
                      style={{
                        marginTop: '24px',
                        borderLeft: `4px solid ${themeColors[theme].primary}`,
                        backgroundColor: theme === 'dark' ? themeColors[theme].card : themeColors[theme].card,
                        textAlign: 'center',
                      }}
                    >
                      <Text size="lg" weight={600} mb="md" style={{ fontFamily: themeColors[theme].headingFont }}>
                        Welcome to Scientific Document QA
                      </Text>
                      <Text mb="md">
                        Upload a document to create a new chat session or select an existing chat from the sidebar.
                      </Text>
                      
                      <Group position="center" spacing="md">
                        <Button
                          variant="light"
                          onClick={toggleTheme}
                          leftIcon={theme === 'dark' ? <Sun size={16} /> : <MoonStars size={16} />}
                        >
                          Switch to {theme === 'dark' ? 'Light' : 'Dark'} Theme
                        </Button>
                      </Group>
                    </Paper>
                  )}
                </Box>
              )}

              {/* Document visualization and chat sections */}
              {selectedId && !showUpload && (
                <>
                  {/* Document analysis section */}
                  <Box style={{ 
                    display: 'flex', 
                    flexDirection: 'column',
                    height: '100%',
                    overflowY: 'auto',
                  }}>
                    <Text 
                      weight={700} 
                      size="xl" 
                      style={{ 
                        fontFamily: themeColors[theme].headingFont,
                        color: theme === 'dark' ? themeColors[theme].text : themeColors[theme].text,
                        marginBottom: '20px',
                        paddingTop: '20px',
                        position: 'sticky',
                        top: 0,
                        backgroundColor: themeColors[theme].background,
                        zIndex: 10,
                      }}
                    >
                      Document Visualization {(showCombinedView || getChatDocuments(selectedChatId).length > 1) && '(Combined)'}
                    </Text>
                    <DocumentAnalysis 
                      theme={theme} 
                      docDetails={showCombinedView ? combinedDocDetails : docDetails} 
                      summary={summary} 
                      loading={loadingDoc} 
                      onDownload={handleDownload} 
                    />
                  </Box>
                  
                  {/* LLMQA section */}
                  <Box style={{ 
                    display: 'flex', 
                    flexDirection: 'column',
                    height: '100%',
                    overflowY: 'auto',
                  }}>
                    <Text 
                      weight={700} 
                      size="xl" 
                      style={{ 
                        fontFamily: themeColors[theme].headingFont,
                        color: theme === 'dark' ? themeColors[theme].text : themeColors[theme].text,
                        marginBottom: '20px',
                        paddingTop: '20px',
                        position: 'sticky',
                        top: 0,
                        backgroundColor: themeColors[theme].background,
                        zIndex: 10,
                      }}
                    >
                      LLM Assistant
                    </Text>
                    <LLMQA 
                      theme={theme} 
                      docId={selectedId}
                      chatId={selectedChatId}
                      disabled={!selectedId || loadingDoc}
                      chatMap={chatQAMap}
                      setChatMap={setChatQAMap}
                      documentIds={selectedChatId ? chatSessions.find(c => c.id === selectedChatId)?.documentIds || [] : [selectedId]}
                    />
                  </Box>
                </>
              )}
            </Box>
          </Container>
        </AppShell>
      </ColorSchemeProvider>
    </MantineProvider>
  );
}

function DocumentAnalysis({ theme, docDetails, summary, loading, onDownload }: {
  theme: ColorScheme;
  docDetails: DocDetails | null;
  summary: string;
  loading: boolean;
  onDownload: () => void;
}) {
  const colors = themeColors[theme];
  const [activeTab, setActiveTab] = useState<string | null>('charts');
  
  if (loading) return (
    <Paper 
      shadow="sm" 
      p="lg" 
      mb="md" 
      radius="lg"
      style={{ 
        fontFamily: colors.fontFamily,
        borderLeft: `4px solid ${colors.primary}`,
        backgroundColor: theme === 'dark' ? colors.card : colors.card,
        display: 'flex',
        justifyContent: 'center',
        alignItems: 'center',
        minHeight: '300px'
      }}
    >
      <Box style={{ textAlign: 'center' }}>
        <Loader size="lg" />
        <Text mt="md">Loading document data...</Text>
      </Box>
    </Paper>
  );
  
  if (!docDetails) return (
    <Paper 
      shadow="sm" 
      p="lg" 
      mb="md" 
      radius="lg"
      style={{ 
        fontFamily: colors.fontFamily,
        borderLeft: `4px solid ${colors.primary}`,
        backgroundColor: theme === 'dark' ? colors.card : colors.card,
        display: 'flex',
        justifyContent: 'center',
        alignItems: 'center',
        minHeight: '300px'
      }}
    >
      <Text color="dimmed">No document data available. The document may still be processing.</Text>
    </Paper>
  );
  
  const chunks = docDetails.chunks || [];
  const fileInfo = docDetails.file_info || {};
  
  // Prepare data for charts
  const chunkData = chunks.map((chunk, i) => ({
    name: `Chunk ${i + 1}`,
    Words: chunk.word_count || 0,
    Characters: chunk.char_count || 0,
    SummaryLength: (chunk.summary || '').length,
  }));
  
  const pieData = chunks.map((chunk, i) => ({
    name: `Chunk ${i + 1}`,
    value: chunk.word_count || 0,
  }));
  
  const COLORS = ['#4361ee', '#3a0ca3', '#4cc9f0', '#7209b7', '#f72585', '#4895ef', '#560bad', '#b5179e'];
  
  return (
    <Paper 
      shadow="sm" 
      p="lg" 
      mb="md" 
      radius="lg"
      style={{ 
        fontFamily: colors.fontFamily,
        borderLeft: `4px solid ${colors.primary}`,
        backgroundColor: theme === 'dark' ? colors.card : colors.card,
      }}
    >
      <Group position="apart" mb="lg">
        <Text 
          weight={700} 
          size="lg"
          style={{ 
            fontFamily: colors.headingFont, 
            fontSize: 22,
            color: theme === 'dark' ? colors.text : colors.text,
          }}
        >
          Document Analysis
        </Text>
        <Button 
          leftIcon={<Download size={16} />} 
          onClick={onDownload} 
          variant="light" 
          size="sm"
          radius="md"
          color={theme === 'dark' ? 'cyan' : 'blue'}
        >
          Download Summary
        </Button>
      </Group>
      
      <Paper 
        p="md" 
        radius="md" 
        withBorder 
        mb="md"
        style={{
          borderColor: theme === 'dark' ? colors.border : colors.border,
          backgroundColor: theme === 'dark' ? 'rgba(255,255,255,0.03)' : 'rgba(0,0,0,0.01)',
        }}
      >
        <Group position="apart">
          <Text weight={600} style={{ fontFamily: colors.headingFont }}>
            {fileInfo.filename || docDetails.filename || 'Document'}
          </Text>
          <Group spacing="md">
            <Badge radius="md" size="md">Chunks: {chunks.length}</Badge>
            <Badge radius="md" size="md">Words: {chunks.reduce((acc, c) => acc + (c.word_count || 0), 0)}</Badge>
            <Badge radius="md" size="md">Avg: {chunks.length ? Math.round(chunks.reduce((acc, c) => acc + (c.word_count || 0), 0) / chunks.length) : 0} words/chunk</Badge>
          </Group>
        </Group>
      </Paper>
      
      <Tabs 
        value={activeTab} 
        onTabChange={setActiveTab}
        radius="md"
        mb="md"
        styles={{
          tabLabel: {
            fontFamily: colors.fontFamily,
            fontWeight: 600,
          },
        }}
      >
        <Tabs.List>
          <Tabs.Tab value="charts">Visualizations</Tabs.Tab>
          <Tabs.Tab value="chunks">Document Chunks</Tabs.Tab>
          <Tabs.Tab value="summary">Summary</Tabs.Tab>
        </Tabs.List>
        
        <Tabs.Panel value="charts" pt="md">
          {loading ? <Loader /> : (
            <>
              <Group grow mb="md">
                <Paper 
                  p="md" 
                  radius="md" 
                  withBorder 
                  style={{ 
                    minWidth: 0,
                    borderColor: theme === 'dark' ? colors.border : colors.border,
                    backgroundColor: theme === 'dark' ? 'rgba(255,255,255,0.03)' : 'rgba(0,0,0,0.01)',
                  }}
                >
                  <Text size="sm" weight={600} mb="md" style={{ fontFamily: colors.headingFont }}>Word Count by Chunk</Text>
                  <ResponsiveContainer width="100%" height={220}>
                    <BarChart data={chunkData} margin={{ top: 10, right: 10, left: 0, bottom: 10 }}>
                      <CartesianGrid strokeDasharray="3 3" opacity={0.1} />
                      <XAxis dataKey="name" fontSize={12} tick={{ fill: theme === 'dark' ? '#e0e0e0' : '#333' }} />
                      <YAxis fontSize={12} tick={{ fill: theme === 'dark' ? '#e0e0e0' : '#333' }} />
                      <RechartsTooltip 
                        contentStyle={{ 
                          backgroundColor: theme === 'dark' ? '#1e293b' : '#fff',
                          borderColor: theme === 'dark' ? '#334155' : '#eaeaea',
                          borderRadius: '8px',
                          fontFamily: colors.fontFamily,
                        }}
                      />
                      <Bar 
                        dataKey="Words" 
                        fill={theme === 'dark' ? '#4cc9f0' : '#4361ee'} 
                        radius={[4, 4, 0, 0]} 
                        animationDuration={1500}
                      />
                    </BarChart>
                  </ResponsiveContainer>
                </Paper>
                <Paper 
                  p="md" 
                  radius="md" 
                  withBorder 
                  style={{ 
                    minWidth: 0,
                    borderColor: theme === 'dark' ? colors.border : colors.border,
                    backgroundColor: theme === 'dark' ? 'rgba(255,255,255,0.03)' : 'rgba(0,0,0,0.01)',
                  }}
                >
                  <Text size="sm" weight={600} mb="md" style={{ fontFamily: colors.headingFont }}>Chunk Distribution</Text>
                  <ResponsiveContainer width="100%" height={220}>
                    <PieChart>
                      <Pie 
                        data={pieData} 
                        dataKey="value" 
                        nameKey="name" 
                        cx="50%" 
                        cy="50%" 
                        outerRadius={80} 
                        label 
                        animationDuration={1500}
                      >
                        {pieData.map((entry, index) => (
                          <Cell 
                            key={`cell-${index}`} 
                            fill={COLORS[index % COLORS.length]} 
                          />
                        ))}
                      </Pie>
                      <RechartsTooltip 
                        contentStyle={{ 
                          backgroundColor: theme === 'dark' ? '#1e293b' : '#fff',
                          borderColor: theme === 'dark' ? '#334155' : '#eaeaea',
                          borderRadius: '8px',
                          fontFamily: colors.fontFamily,
                        }}
                      />
                      <Legend 
                        verticalAlign="bottom" 
                        height={36} 
                        formatter={(value, entry, index) => (
                          <span style={{ color: theme === 'dark' ? '#e0e0e0' : '#333', fontFamily: colors.fontFamily }}>
                            {value}
                          </span>
                        )}
                      />
                    </PieChart>
                  </ResponsiveContainer>
                </Paper>
              </Group>
              <Paper 
                p="md" 
                radius="md" 
                withBorder 
                mb="md"
                style={{
                  borderColor: theme === 'dark' ? colors.border : colors.border,
                  backgroundColor: theme === 'dark' ? 'rgba(255,255,255,0.03)' : 'rgba(0,0,0,0.01)',
                }}
              >
                <Text size="sm" weight={600} mb="md" style={{ fontFamily: colors.headingFont }}>Summary Length vs. Chunk Length</Text>
                <ResponsiveContainer width="100%" height={220}>
                  <ScatterChart margin={{ top: 10, right: 10, left: 10, bottom: 10 }}>
                    <CartesianGrid strokeDasharray="3 3" opacity={0.1} />
                    <XAxis 
                      dataKey="Words" 
                      name="Words" 
                      fontSize={12} 
                      tick={{ fill: theme === 'dark' ? '#e0e0e0' : '#333' }}
                      label={{ 
                        value: 'Word Count', 
                        position: 'bottom', 
                        fill: theme === 'dark' ? '#e0e0e0' : '#333',
                        fontFamily: colors.fontFamily,
                      }}
                    />
                    <YAxis 
                      dataKey="SummaryLength" 
                      name="Summary Length" 
                      fontSize={12} 
                      tick={{ fill: theme === 'dark' ? '#e0e0e0' : '#333' }}
                      label={{ 
                        value: 'Summary Length', 
                        angle: -90, 
                        position: 'left', 
                        fill: theme === 'dark' ? '#e0e0e0' : '#333',
                        fontFamily: colors.fontFamily,
                      }}
                    />
                    <ZAxis dataKey="Characters" range={[60, 400]} />
                    <RechartsTooltip 
                      cursor={{ strokeDasharray: '3 3' }}
                      contentStyle={{ 
                        backgroundColor: theme === 'dark' ? '#1e293b' : '#fff',
                        borderColor: theme === 'dark' ? '#334155' : '#eaeaea',
                        borderRadius: '8px',
                        fontFamily: colors.fontFamily,
                      }}
                      formatter={(value, name) => [`${value}`, name]}
                    />
                    <Scatter 
                      name="Chunks" 
                      data={chunkData} 
                      fill={theme === 'dark' ? '#4cc9f0' : '#4361ee'}
                      animationDuration={1500}
                    />
                  </ScatterChart>
                </ResponsiveContainer>
              </Paper>
            </>
          )}
        </Tabs.Panel>
        
        <Tabs.Panel value="chunks" pt="md">
          <ScrollArea style={{ height: 350 }}>
            {chunks.map((chunk, i) => (
              <Paper 
                key={i} 
                p="md" 
                mb="md" 
                radius="md" 
                withBorder
                style={{
                  borderColor: theme === 'dark' ? colors.border : colors.border,
                  backgroundColor: theme === 'dark' ? 'rgba(255,255,255,0.03)' : 'rgba(0,0,0,0.01)',
                }}
              >
                <Group position="apart" mb="xs">
                  <Text weight={600} style={{ fontFamily: colors.headingFont }}>Chunk {i + 1}</Text>
                  <Badge size="sm" radius="md">
                    {chunk.word_count || 0} words
                  </Badge>
                </Group>
                <Text size="sm" mb="sm">
                  <Text component="span" weight={600}>Summary:</Text> {chunk.summary}
                </Text>
                <Paper 
                  p="xs" 
                  radius="sm" 
                  style={{
                    backgroundColor: theme === 'dark' ? 'rgba(15,23,42,0.7)' : 'rgba(240,245,255,0.7)',
                    fontFamily: 'monospace',
                    fontSize: '12px',
                    maxHeight: '100px',
                    overflow: 'auto',
                  }}
                >
                  {chunk.excerpt || chunk.text.slice(0, 150) + '...'}
                </Paper>
              </Paper>
            ))}
          </ScrollArea>
        </Tabs.Panel>
        
        <Tabs.Panel value="summary" pt="md">
          {summary ? (
            <Paper 
              p="md" 
              radius="md" 
              withBorder
              style={{
                backgroundColor: theme === 'dark' ? 'rgba(255,255,255,0.03)' : 'rgba(0,0,0,0.01)',
                borderColor: theme === 'dark' ? colors.border : colors.border,
                minHeight: '300px',
              }}
            >
              <Text size="sm" lineHeight={1.7} style={{ whiteSpace: 'pre-wrap' }}>
                {summary}
              </Text>
            </Paper>
          ) : (
            <Text color="dimmed" align="center" my="xl">No summary available for this document.</Text>
          )}
        </Tabs.Panel>
      </Tabs>
    </Paper>
  );
}

function LLMQA({ theme, docId, chatId, disabled, chatMap, setChatMap, documentIds }: { 
  theme: ColorScheme; 
  docId: string | null; 
  chatId: string | null;
  disabled: boolean;
  chatMap: Record<string, { question: string; answer: string; sources?: any[] }[]>;
  setChatMap: React.Dispatch<React.SetStateAction<Record<string, { question: string; answer: string; sources?: any[] }[]>>>;
  documentIds: string[];
}) {
  const colors = themeColors[theme];
  const [question, setQuestion] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const answerRef = useRef<HTMLDivElement | null>(null);
  const [activeSource, setActiveSource] = useState<number | null>(null);

  // Get the QA history for the current chat or document
  const qaHistory = chatId ? (chatMap[chatId] || []) : (docId ? chatMap[docId] || [] : []);

  useEffect(() => {
    answerRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [qaHistory]);

  const handleAsk = async () => {
    if (!question.trim() || (!docId && documentIds.length === 0)) return;
    setError(null);
    setIsLoading(true);
    
    try {
      // If in a chat with multiple documents, use first document for now
      // In a real implementation, you would query against all documents or have a backend endpoint for multi-document queries
      const targetDocId = docId || documentIds[0];
      
      const res = await axios.post(`${BACKEND_URL}/api/answer`, {
        question,
        document_id: targetDocId
      });
      
      const newEntry = { 
        question, 
        answer: res.data.answer || 'No answer.', 
        sources: res.data.sources 
      };
      
      // Update the chat map
      const chatKey = chatId || docId || 'default';
      setChatMap(prev => ({
        ...prev,
        [chatKey]: [...(prev[chatKey] || []), newEntry]
      }));
      
      setQuestion('');
    } catch (err: any) {
      setError(err?.response?.data?.error || 'Failed to get answer.');
    } finally {
      setIsLoading(false);
    }
  };

  const handleClear = () => {
    const chatKey = chatId || docId || 'default';
    setChatMap(prev => ({
      ...prev,
      [chatKey]: []
    }));
  };

  return (
    <Paper 
      shadow="sm" 
      p="lg" 
      radius="lg"
      style={{ 
        fontFamily: colors.fontFamily,
        borderLeft: `4px solid ${colors.primary}`,
        backgroundColor: theme === 'dark' ? colors.card : colors.card,
        position: 'relative',
        display: 'flex',
        flexDirection: 'column',
        width: '100%',
        height: '100%',
      }}
    >
      {documentIds.length > 1 && (
        <Box mb="md">
          <Text size="sm" weight={600}>
            Using multiple documents ({documentIds.length})
          </Text>
        </Box>
      )}
      
      <Box style={{ flex: 1, display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
        <Group position="apart" mb="xs">
          <Text weight={600} style={{ fontFamily: colors.headingFont }}>Chat History</Text>
          <Button size="xs" color="red" variant="light" onClick={handleClear} disabled={qaHistory.length === 0}>Clear</Button>
        </Group>
        <ScrollArea style={{ flex: 1, minHeight: 0 }}>
          {qaHistory.length === 0 ? (
            <Text color="dimmed" size="sm">No questions asked yet.</Text>
          ) : (
            qaHistory.map((qa, idx) => (
              <Paper key={idx} p="sm" mb="sm" radius="md" withBorder style={{ background: theme === 'dark' ? '#232946' : '#f8fafc' }}>
                <Text size="sm" weight={600} mb={4} style={{ color: colors.primary }}>Q: {qa.question}</Text>
                <Text size="sm" style={{ color: colors.text, whiteSpace: 'pre-wrap' }} mb={8}>
                  A: {qa.answer}
                </Text>
                
                {qa.sources && qa.sources.length > 0 && (
                  <Box mt={8}>
                    <Text size="xs" weight={600} mb={4} color="dimmed">Sources:</Text>
                    <Group spacing={8}>
                      {qa.sources.map((source, sourceIdx) => (
                        <Popover 
                          key={sourceIdx} 
                          width={300} 
                          position="top" 
                          withArrow 
                          shadow="md"
                          opened={activeSource === sourceIdx + (idx * 100)}
                          onClose={() => setActiveSource(null)}
                        >
                          <Popover.Target>
                            <Badge 
                              size="sm"
                              style={{ cursor: 'pointer' }}
                              onMouseEnter={() => setActiveSource(sourceIdx + (idx * 100))}
                              onMouseLeave={() => setActiveSource(null)}
                            >
                              Chunk {source.chunk_index || sourceIdx + 1}
                            </Badge>
                          </Popover.Target>
                          <Popover.Dropdown>
                            <Text size="xs" weight={600} mb={4}>Relevance Score:</Text>
                            <Text size="sm" color={
                              (source.relevance_score || 0) > 0.7 ? 'green' : 
                              (source.relevance_score || 0) > 0.4 ? 'orange' : 'red'
                            }>
                              {source.relevance_score?.toFixed(3) || 'N/A'}
                            </Text>
                            
                            {source.text && (
                              <>
                                <Text size="xs" weight={600} mt={8} mb={4}>Text Excerpt:</Text>
                                <Paper 
                                  p="xs" 
                                  style={{
                                    maxHeight: '120px',
                                    overflow: 'auto',
                                    backgroundColor: theme === 'dark' ? 'rgba(15,23,42,0.7)' : 'rgba(240,245,255,0.7)',
                                    fontFamily: 'monospace',
                                    fontSize: '12px',
                                    whiteSpace: 'pre-wrap',
                                  }}
                                >
                                  {source.text.substring(0, 300)}...
                                </Paper>
                              </>
                            )}
                          </Popover.Dropdown>
                        </Popover>
                      ))}
                    </Group>
                  </Box>
                )}
              </Paper>
            ))
          )}
          <div ref={answerRef} />
        </ScrollArea>
      </Box>
      <Divider my="md" />
      <Group align="flex-end" spacing="md">
        <TextInput
          placeholder="Type your question..."
          value={question}
          onChange={e => setQuestion(e.currentTarget.value)}
          disabled={disabled || isLoading}
          style={{ flex: 1, fontFamily: colors.fontFamily }}
          size="md"
        />
        <Button
          leftIcon={<Send size={16} />}
          onClick={handleAsk}
          loading={isLoading}
          disabled={disabled || isLoading || !question.trim()}
          size="md"
          style={{ fontFamily: colors.fontFamily }}
        >
          Ask
        </Button>
      </Group>
      {error && <Notification color="red" mt="md" withCloseButton={false}>{error}</Notification>}
    </Paper>
  );
}