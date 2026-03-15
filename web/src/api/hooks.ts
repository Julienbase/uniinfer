import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import {
  fetchStatus,
  fetchDevices,
  fetchCachedModels,
  fetchAliases,
  deleteModel,
  fetchChatSessions,
  fetchChatSession,
  fetchRecentMessages,
} from "./client";

export function useStatus(refetchInterval = 5000) {
  return useQuery({
    queryKey: ["status"],
    queryFn: fetchStatus,
    refetchInterval,
  });
}

export function useDevices() {
  return useQuery({
    queryKey: ["devices"],
    queryFn: fetchDevices,
    select: (data) => data.devices,
  });
}

export function useCachedModels() {
  return useQuery({
    queryKey: ["cached-models"],
    queryFn: fetchCachedModels,
    select: (data) => data.models,
  });
}

export function useAliases() {
  return useQuery({
    queryKey: ["aliases"],
    queryFn: fetchAliases,
    select: (data) => data.aliases,
  });
}

export function useDeleteModel() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: ({ modelId, quantization, format = "gguf" }: { modelId: string; quantization: string; format?: string }) =>
      deleteModel(modelId, quantization, format),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["cached-models"] });
      queryClient.invalidateQueries({ queryKey: ["aliases"] });
    },
  });
}

export function useChatSessions(refetchInterval = 5000) {
  return useQuery({
    queryKey: ["chat-sessions"],
    queryFn: fetchChatSessions,
    select: (data) => data.sessions,
    refetchInterval,
  });
}

export function useChatSession(sessionId: string | null) {
  return useQuery({
    queryKey: ["chat-session", sessionId],
    queryFn: () => fetchChatSession(sessionId!),
    enabled: !!sessionId,
    refetchInterval: 3000,
  });
}

export function useRecentMessages(limit = 50) {
  return useQuery({
    queryKey: ["recent-messages", limit],
    queryFn: () => fetchRecentMessages(limit),
    select: (data) => data.messages,
    refetchInterval: 3000,
  });
}
